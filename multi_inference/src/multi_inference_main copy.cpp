#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <thread>
#include <queue>
#include <c10/cuda/CUDAFunctions.h>
#include <opencv2/opencv.hpp>
#include <string>  //결과 출력용(output result)
#include <chrono>  //시간 측정용(time check)

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ./loadgen 폴더의 헤더함수들(header  in Folder loadgen)
#include "loadgen.h" 
#include "query_sample.h"
#include "test_settings.h"
#include "system_under_test.h"
#include "query_sample_library.h"
#include "A.h"
#include "Inferencer.h"

#define DEBUG 0
#define MYSERVER 1 //0은 고대 서버

//Inferencer들의 시작과 중지 제어변수(Start and Stop Control Variable for SUT Threads)
static std::atomic<bool> inferencers_keepalive_0;
static std::atomic<bool> inferencers_keepalive_1; 

std::vector<at::cuda::CUDAStream> streams;

/*lg 스레드를 제어하기 위한 bsem(to control lg thread)*/
/* Binary semaphore */
typedef struct bsem {
	pthread_mutex_t mutex;
	pthread_cond_t   cond;
	int v;
} bsem;
/* Init semaphore to 1 or 0 */
static void bsem_init(bsem *bsem_p, int value) {
	if (value < 0 || value > 1) {
		exit(1);
	}
	pthread_mutex_init(&(bsem_p->mutex), NULL);
	pthread_cond_init(&(bsem_p->cond), NULL);
	bsem_p->v = value;
}

/* Reset semaphore to 0 */
static void bsem_reset(bsem *bsem_p) {
	bsem_init(bsem_p, 0);
}

/* Post to at least one thread */
static void bsem_post(bsem *bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	bsem_p->v = 1;
	pthread_cond_signal(&bsem_p->cond);
	pthread_mutex_unlock(&bsem_p->mutex);
}
/* Wait on semaphore until semaphore has value 0 */
static void bsem_wait(bsem* bsem_p) {
	pthread_mutex_lock(&bsem_p->mutex);
	pthread_cond_wait(&bsem_p->cond, &bsem_p->mutex);
  while (bsem_p->v != 1) {}
	bsem_p->v = 0;
	pthread_mutex_unlock(&bsem_p->mutex);
}

/*응답큐: SUT가 쿼리를 처리한 후, loadgen에게 보내는 응답을 담는 큐*/
class responseQueue {
public:
    mlperf::QuerySampleResponse *front;
    mlperf::QuerySampleResponse *rear;
    pthread_mutex_t rwmutex; 
    bsem *has_jobs;   
    int len;
    responseQueue() {
        front = NULL;
        rear = NULL;
        len = 0;
        has_jobs = (struct bsem*)malloc(sizeof(struct bsem));
        bsem_init(has_jobs, 0);
        pthread_mutex_init(&rwmutex, NULL);
    }
    void push(mlperf::QuerySampleResponse *newqsr) {
        pthread_mutex_lock(&rwmutex);
        switch(len) {
            case 0:  
                front = newqsr;
                rear = newqsr;
                break; 
            default: 
                rear->prev = newqsr;
                rear = newqsr;
        }
        len++;
        pthread_mutex_unlock(&rwmutex);
        bsem_post(has_jobs);  /*lg 스레드의 대기를 종료하는 신호*/
    }
    mlperf::QuerySampleResponse* pop() {
        pthread_mutex_lock(&rwmutex);
        mlperf::QuerySampleResponse *qsr_f = front;
        switch(len) {
            case 0:
              qsr_f = NULL;
                break;
            case 1:
                front = NULL;
                rear = NULL;
                len = 0;
                break;
            default: 
                front = qsr_f->prev;
                len--;
        }
        pthread_mutex_unlock(&rwmutex);
        return qsr_f;
    }
    /*디버깅을 위한 출력 함수*/
    void rq_print(){
    pthread_mutex_lock(&rwmutex);
    if (len == 0) {
        std::cout << "[null queue]\n";
    } else {
        std::cout << "[queue elements:";
        mlperf::QuerySampleResponse* qsr = front;
        while (qsr != NULL) {
            std::cout << qsr->id << " ";
            qsr = qsr->prev;
        }
        std::cout << "]\n";
    }
    pthread_mutex_unlock(&rwmutex);
    }
    void destroy() {
        front = NULL;
        rear = NULL;
        len = 0;
        bsem_reset(has_jobs);
        free(has_jobs);
    }
};
/*버퍼 큐: lg로부터 받은 쿼리를 담으며 Inferencer들이 하나씩 가지고 있음*/
template <typename T>
class BufferedQueue {
 public:
  BufferedQueue(const size_t max_items)
      : kTimeOutMs_(1), kMaxItems_(max_items) {}
  void Push(const T& value) {
    std::unique_lock<std::mutex> lk(queue_m_);
    queue_.push(value);
    queue_cv_.notify_one();
  }
  std::vector<T> Pop() {
    using namespace std::chrono_literals;  // NOLINT
    std::unique_lock<std::mutex> lk(queue_m_);
    queue_cv_.wait_for(lk, kTimeOutMs_ * 1ms, [this] {
      return (queue_.size() >= kMaxItems_);
    });
    const size_t kPopSize = std::min(queue_.size(), kMaxItems_);
    std::vector<T> result(kPopSize);
    for (size_t repeat = 0; repeat < kPopSize; repeat++) {
      result[repeat] = queue_.front();
      queue_.pop();
    }
    return result;
  }
 private:
  const size_t kTimeOutMs_;
  const size_t kMaxItems_;
  std::queue<T> queue_;
  std::mutex queue_m_;
  std::condition_variable queue_cv_;
};
/*args로 받은 주소값을 분리하기 위한 split함수*/
std::vector<std::string> SplitE1(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;
  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }
  return answer;
};

/*class about one model inferencer*/
class Inferencer
{
private:
    torch::jit::script::Module m;
    pthread_t thread_;
    int id;  
    BufferedQueue<mlperf::QuerySample>* issued_query_samples_buffered_ = nullptr; 
    std::vector<std::string> total_input_filenames_;  /*데이터 셋의 파일 이름 저장 벡터*/
    std::vector<int> total_expected_labels_;          /*데이터 셋의 라벨 저장 벡터*/
    int8_t* image_sequence_;  //데이터 셋의 데이터들을 가리키는 포인터
    const size_t kNumOfProcessingThread_ = 6;
    const size_t kMaxBatch = 18;
    const size_t image_size_ = 224 * 224 * 3;
    
public:
    std::vector<cv::Mat> images;
    std::vector<int8_t*> data_ptrs_int8;
    responseQueue *send_lg_response_queue_;  //Inferencer들이 공유하는 응답큐
    std::string dataset_path_;
    std::string model_name_;
    size_t total_sample_count_;
    size_t performance_sample_count_;
    mlperf::TestMode mlperf_mode_;
    int result_good=0;
    int result_total=0;
    std::vector<double> result_timing;   /*추론 시간을 저장하는 벡터*/
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> last_loaded; /*데이터셋에 접근한 최종시간*/
    bool inferencers_keepalive;
    Inferencer(){inferencers_keepalive=true; };
    /*클래스의 응답큐 초기화. SUT안의 Inferencer들이 공유하는 큐이기 때문에 포인터*/
    void init_responseQ(responseQueue* rq)
    {
      send_lg_response_queue_=rq;
    }
    /*클래스 초기화 함수. model load 및 dataset내의 파일이름 저장, 클래스 변수 초기화 */
    int Init(const char* model_path,char*& dataset_path,const char* model_name,int id) //,char*& Annotation_path
    {
        // c10::DeviceIndex GPU_NUM = 2;
        // c10::cuda::set_device(GPU_NUM);
        // torch::Device device = {at::kCUDA,GPU_NUM};
        
        this->id=id;
        issued_query_samples_buffered_ = new BufferedQueue<mlperf::QuerySample>(kMaxBatch);
        dataset_path_=dataset_path;
        
        #if MYSERVER
          std::ifstream openFile("/home/seoyeon/seoyeon_big/inference/multiSUT/fake_imagenet/val_map.txt");
        #else
          std::ifstream openFile("/home/kmsguest/big-stroage/seoyeon/mlperf/inference/multiSUT/fake_imagenet/val_map.txt");
        //kAnnotationPath.data()
        #endif

        if (openFile.is_open()) {
            std::string line;
            while (getline(openFile, line)) {
              std::vector<std::string> result = SplitE1(line, ' '); /*valmap파일을 한줄씩 읽어서 분리 ex) val/800px-Porsche_991_silver_IAA.jpg 817*/
              std::string filename = dataset_path_+"/"+result[0];
              total_expected_labels_.push_back(stoi(result[1]));
              total_input_filenames_.push_back(filename);
            }
            openFile.close();
        }else
          std::cout<<"is not open\n";
        total_sample_count_ = total_input_filenames_.size();
        performance_sample_count_ = total_sample_count_;
        model_name_ = model_name;

        /*모델 로드 및 gpu에 올려두기, 테스트*/
        at::Device device(torch::kCUDA);
        try
        {
            std::cout<<model_path<<'\n';
            m= torch::jit::load(model_path);
            m.eval();
            m.to(device); 
            
            //warm up
            at::Tensor input_tensor = torch::randn({1,224, 224, 3});
            std::vector<torch::jit::IValue> inputs;
            input_tensor.set_requires_grad(false);
            input_tensor = input_tensor.to(device, torch::kFloat);
            input_tensor = input_tensor.permute({0,3,1,2});  
            inputs.push_back(input_tensor); 
            m.forward({input_tensor}).toTensor();
        }
        catch(const std::exception& e)
        {
            std::cerr <<"model load fail\n"<<e.what() << '\n';
        }
        return 1; 
    };
    /*static function: 스레드 함수*/
    /*자신의 버퍼큐에 쿼리 삽입시, 꺼내서 추론하고, 다음 쿼리를 받기 위해 응답큐에 응답 삽입*/
    static void* model_do(Inferencer *inferencer)
    {
      at::Device device(torch::kCUDA);
      std::vector<torch::jit::IValue> inputs;
      at::cuda::CUDAStreamGuard guard(streams[inferencer->id]);
      #if DEBUG
      #else
      #endif
        while(inferencer->inferencers_keepalive) 
        {
            auto buffered_inputs = inferencer->issued_query_samples_buffered_->Pop();
            if(buffered_inputs.size()>0){
                std::cout<<"buffered_size: "<<buffered_inputs.size()<<"\n"; 
                cudaEvent_t start_CUDA, end_CUDA;
                float time;
                auto now1 = std::chrono::high_resolution_clock::now();
                auto duration1 = std::chrono::duration<double, std::milli>(now1 - inferencer->last_loaded);
                std::cout<<"[id-"<<inferencer->id<<" ]:model_do, "<<duration1.count()*0.001<<"s  \n";
                cudaEventCreate(&start_CUDA);
                cudaEventCreate(&end_CUDA);
                cudaEventRecord(start_CUDA);
                int max_idx[inferencer->kMaxBatch];
                struct mlperf::QuerySampleResponse responses[inferencer->kMaxBatch];
                struct mlperf::QuerySampleResponse *responses2q = new mlperf::QuerySampleResponse[inferencer->kMaxBatch];
                auto start = std::chrono::high_resolution_clock::now(); 
                for (size_t idx = 0; idx < buffered_inputs.size(); ++idx) {
                    const mlperf::QuerySample& query_sample = buffered_inputs[idx];
                    std::cout<<"[id-"<<inferencer->id<<" ]"<<"sample: "<<query_sample.DNNindex<<" "<<query_sample.index<<"\n";
                    inputs.clear();
                    at::Tensor input_tensor = torch::from_blob(inferencer->images[query_sample.index].data, {1, 224,224,3},torch::kFloat32);
                    input_tensor = input_tensor.permute({0,3,1,2}); 
                    input_tensor = input_tensor.to(device,torch::kFloat32);
                    at::Tensor result = inferencer->m.forward({input_tensor}).toTensor();
                    result = result.to(torch::kCPU);
                    auto max_result = torch::argmax(result,1);
                    max_idx[idx] = max_result.item<int>();
                    //std::cout<<max_result.item<int>()<<"\n";
                    //std::cout<<"결과: "<<max_idx[idx]<<" : "<<inferencer->total_expected_labels_[query_sample.index]<<"\n";
                    if(max_idx[idx]==inferencer->total_expected_labels_[query_sample.index])
                      inferencer->result_good++;
                    inferencer->result_total++;
                }
                /*후처리*/
                cudaStreamSynchronize(streams[inferencer->id]);
                cudaEventRecord(end_CUDA);
                cudaEventSynchronize(end_CUDA);
                cudaEventElapsedTime(&time, start_CUDA, end_CUDA);
                // std::cout<<"[id-"<<inferencer->id<<" ]"<<" result  "<<time/1000.0<<"s  \n";
                auto later = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double, std::milli>(later - start);
                std::cout<<"[id-"<<inferencer->id<<" ]"<<" result  "<<duration.count()*0.001<<"s ***** \n";
                inferencer->result_timing.push_back(duration.count()*0.001);
                std::vector<std::array<uint8_t, sizeof(float)>> post_results(buffered_inputs.size());
                uintptr_t bi[buffered_inputs.size()];
                for (int i = 0; i < buffered_inputs.size(); i++) {
                    std::memcpy(post_results[i].data(), &max_idx[i], sizeof(int)); //float
                    bi[i] = reinterpret_cast<uintptr_t>(post_results[i].data());
                }
                
                for (size_t i = 0; i < buffered_inputs.size(); i++) {
                    responses[i].id = buffered_inputs[i].id;
                    responses[i].data = bi[i];
                    responses[i].size = 4;
                    responses[i].DNNindex=buffered_inputs[i].DNNindex;
                    responses2q[i].id = buffered_inputs[i].id;
                    responses2q[i].data = bi[i];
                    responses2q[i].size = 4;
                    responses2q[i].DNNindex=buffered_inputs[i].DNNindex;
                }
                inferencer->send_lg_response_queue_->push(responses2q);
                QuerySamplesResponseLogging(responses,buffered_inputs.size());
            }
        }
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(now - inferencer->last_loaded);
        inferencer->add_results("TestScenario.SingleStream",inferencer->model_name_
                                ,inferencer->result_timing,duration.count()*0.001,true);
        inferencer->end();
        return NULL;
    };
    void join()
    {
        pthread_join(thread_,NULL);  //join의 사용법이 잘못되었을 수도 있음 체크
    }
    void end()
    {
        std::cout <<id<<": END" << std::endl;
    };
    void start()
    {
        pthread_create(&thread_,NULL,(void*(*)(void*))model_do,this);
    }
    void SutIssueQuery(const std::vector<mlperf::QuerySample>& samples)
    {
        for (auto sample : samples) {
            
            issued_query_samples_buffered_->Push(sample);
        }
    };
    void SutFlushQueries(){};
    cv::Mat resize_with_aspectratio(cv::Mat img, int out_height, int out_width, double scale=87.5, int inter_pol=cv::INTER_LINEAR) {
        int height = img.rows;
        int width = img.cols;
        int new_height = static_cast<int>(100.0 * out_height / scale);
        int new_width = static_cast<int>(100.0 * out_width / scale);

        int h, w;
        if (height > width) {
            w = new_width;
            h = static_cast<int>(new_height * height / width);
        } else {
            h = new_height;
            w = static_cast<int>(new_width * width / height);
        }
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(w, h), 0, 0, inter_pol);
        return resized_img;
    }
  cv::Mat center_crop(cv::Mat img, int out_height, int out_width) {
      int height = img.rows;
      int width = img.cols;
      int left = (width - out_width) / 2;
      int right = (width + out_width) / 2;
      int top = (height - out_height) / 2;
      int bottom = (height + out_height) / 2;
      cv::Rect roi(left, top, out_width, out_height);
      cv::Mat img_cropped = img(roi);
      return img_cropped;
  }

    cv::Mat pre_process_img(cv::Mat img,int i=0) 
    {
      cv::Mat img_rgb;
      cv::Size dims=cv::Size(224, 224);
      cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
      int output_height = dims.height;
      int output_width = dims.width;
      cv::Mat img_resized;
      img_resized = resize_with_aspectratio(img_rgb,dims.height,dims.width,87.5,cv::INTER_AREA);
      cv::Mat img_cropped;
      img_cropped = center_crop(img_resized,dims.height,dims.width);
      cv::Mat img_float;
      img_cropped.convertTo(img_float, CV_32FC3);
      cv::Mat means(224, 224, CV_32FC3, cv::Scalar(123.68, 116.78, 103.94));
      cv::subtract(img_float, means, img_float);
      return img_float;
  }

    void QslLoadSamplesToRam(
        const std::vector<mlperf::QuerySampleIndex>& samples){
        int i=0;
        for (const auto& filename : total_input_filenames_) {
            cv::Mat img = cv::imread(filename);
            cv::Mat pre_proc_img;
            pre_proc_img = pre_process_img(img);
            // 이미지를 0~1 범위의 값으로 정규화합니다.
            images.push_back(pre_proc_img);
        }
        last_loaded = std::chrono::high_resolution_clock::now();
        std::cout<<"load done\n";
    };
    void QslUnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex>& samples){
        free(image_sequence_);
    };
    double getMean(std::vector<double> const&vec)
    {
      double sum = 0;
      for(int i=0;i<vec.size();i++)
      {
        sum+= (double) vec.at(i); 
      }
      return sum / vec.size();
    }
    /*inference 결과를 출력*/
    void add_results(const char* name,std::string model,std::vector<double> result_list,double took,bool show_accuracy=false)//final_results[],const char* name,const char* model,result_dict[],result_list[],time_t took ,bool show_accuracy=false)
    {
      std::map<std::string, double> result;
      std::string result_str="";

      /*추론 시간 결과들을 percentiles를 따져서 저장*/
      std::vector<double> percentiles = {50.0, 80.0, 90.0, 95.0, 99.0, 99.9};
      std::sort(result_list.begin(), result_list.end());
      std::vector<double> buckets;

      for (double percentile : percentiles) {
          int index = (int)percentile / 100.0 * (result_list.size() - 1);
          double bucket = result_list[index];
          buckets.push_back(bucket);
      }
      if(result_total == 0)
            result_total = result_list.size();


      double qps = result_list.size()/took;
      double q_latency = took/result_list.size();
      double mean = getMean(result_list);

      float accuracy;
      if(show_accuracy)
      {
        result["accuracy"] = 100.0 * result_good / result_total;
        accuracy = result["accuracy"];
      }
      result_str+= "[" + std::to_string(id) + "] " + name + " Model= " + model + ", qps=" + std::to_string(qps) + ", 1-query=" + std::to_string(q_latency) +
       " mean=" + std::to_string(mean) + ", time=" + std::to_string(took) +
       ", acc=" + std::to_string(accuracy) + "%, queries=" + std::to_string(result_list.size());
      std::cout<<result_str<<"\n";

    }

}; //class Inferencer


class SystemUnderTestMulti : public mlperf::SystemUnderTest {
 private:
  Inferencer(* inferencers)[2];
  //모델 클래스  
  std::string name_{"MultiSUT"};
  responseQueue * responseQ;
 public:
  SystemUnderTestMulti(Inferencer(* i)[2]): inferencers(i){
    responseQ=new responseQueue();
    
    (*inferencers)[0].init_responseQ(responseQ);
    (*inferencers)[1].init_responseQ(responseQ);
  }
  ~SystemUnderTestMulti() override = default;
  const std::string& Name() override { return name_; }
  /*lg 스레드가 해당 함수 호출. 맞는 inferencer에 쿼리 보냄*/
  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {
    mlperf::QueryReceiveComplete(samples[0].id); /*기존 lg의 대기를 바로 종료시키기 위한 장치 */
    if(samples[0].DNNindex==0)
      (*inferencers)[0].SutIssueQuery(samples);
    else
      (*inferencers)[1].SutIssueQuery(samples);
  }
  void FlushQueries() override { (*inferencers)[0].SutFlushQueries(); }
  mlperf::QuerySampleResponse* response_pop()
  {
    mlperf::QuerySampleResponse* response=responseQ->pop();
    return response;
  }
  void RQ_wait()
  {
    bsem_wait(responseQ->has_jobs);
  }
  void RQ_print()
  {
    responseQ->rq_print();
  }
  void inferencer_end(int id)
    {
      (*inferencers)[id].inferencers_keepalive=false;
    };
};

class QuerySampleLibraryMulti : public mlperf::QuerySampleLibrary {
 private:
  Inferencer(* inferencers)[2]; //배열 포인터. 배열의 주소를 가지고있음. 포인터 배열과 다름 [2]가리키는 포인터라는 의미
  std::string name_{"MultiQSL"};

 public:
  QuerySampleLibraryMulti(Inferencer(* i)[2]) : inferencers(i){
    //inferencers = i;
  }
  ~QuerySampleLibraryMulti() = default;
  const std::string& Name() override { return name_; }
  size_t TotalSampleCount() override {
    return (*inferencers)[0].total_sample_count_;
  }
  size_t PerformanceSampleCount() override {
    return (*inferencers)[0].performance_sample_count_;
  }
  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    (*inferencers)[0].QslLoadSamplesToRam(samples);
    (*inferencers)[1].QslLoadSamplesToRam(samples);
  }
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    (*inferencers)[0].QslUnloadSamplesFromRam(samples);
  }
};

// void inferencer_end()
// {
//   inferencers_keepalive_0=!inferencers_keepalive_0;
// }

/*디버깅을 위한 테스트 함수. lg의 역할을 함*/
void lg_test2(Inferencer(* inferencers)[2])
{
  while(inferencers_keepalive_0)
  {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto responses=(*inferencers)[0].send_lg_response_queue_->pop();
    if(responses!=NULL){ 
      std::vector<mlperf::QuerySample> query_to_send; 
      query_to_send.push_back({0,0,0});
      (*inferencers)[0].SutIssueQuery(query_to_send);
    }
  }
};
int test_m(char*& model_path)
{
  std::cout<<"!"<<model_path<<"\n";
  //std::string model_path_ = model_path;
  try{
    torch::jit::script::Module module = torch::jit::load(model_path);

  }catch (const c10::Error& e) {
    std::cerr << "Error loading the model: " << e.msg() << std::endl;
    return -1;
  }
  std::cout<<"load done\n";
  return 1;
}



void streamTest(){
  std::cout<<"streamTest()\n";
  for(int i=0; i<2; i++){
    streams.push_back(at::cuda::getStreamFromPool(true,0));
  }
}

int main(int argc,char *argv[])
{
  /*작동과정을 테스트하기 위한 부분*/
  #if DEBUG
    
    std::cout<<"debug mode\n";
    
    std::cout<<"start";
    Inferencer Inferencers[2];
    //std::vector<Inferencer> Inferencers;
    for(int i=0;i<1;i++)
    {
        //Inferencers[i] = new Inferencer(argv[1],i); //객체의 주소값을.
        Inferencers[i].Init("/home/seoyeon/seoyeon_big/inference/multiSUT/resnet50.pt",argv[2],"resnet50",i);
        
    }
    //Inferencers[0].start();

    std::cout<<"start";
    //SystemUnderTestMulti sut(&Inferencers);
    //QuerySampleLibraryMulti qsl(&Inferencers);
    mlperf::TestSettings test_settings;
    mlperf::LogSettings log_settings;
    
    //lg_test2(&Inferencers);
    //x`lg_sut_test(&sut);
    //Inferencers[0].join();
    //Inferencers[1].join();
    std::cout<<"done"<<std::endl;
    
    // for(int i=0;i<2;i++)
    //     delete Inferencers[i];
    return 0;
  #else
    std::cout<<"none debug mode@\n";
    at::cuda::CUDAGuard guard({at::kCUDA,0});
    streamTest();

    int multiN=2;
    inferencers_keepalive_0=true;
    inferencers_keepalive_1=true;
    //test_m(argv[1]);
    //return 1;
    Inferencer Inferencers[2];
    /*inferencer 들 초기화(동적 문제로 인해 현재는 고정 크기의 배열로 설정함)*/
    for(int i=0;i<multiN;i++)
    {
        if(i==0)
          Inferencers[i].Init("/home/seoyeon/seoyeon_big/inference/multi_inference/dnn_models/resnet50.pt",argv[2],"resnet50",i);
          //Inferencers[i].Init("/home/seoyeon/seoyeon_big/inference/multiSUT/alexnet_model.pt",argv[2],"alexnet",i);
        else 
          Inferencers[i].Init("/home/seoyeon/seoyeon_big/inference/multi_inference/dnn_models/resnet50.pt",argv[2],"resnet50",i);
        //Inferencers[i].Init(argv[1],argv[2],argv[3],i);
        Inferencers[i].start();
    }
    //Inferencers[0].start();  //하나만 테스트하고싶을때는 이거 주석풀고, for문 2는 그대로*///

    SystemUnderTestMulti sut(&Inferencers);
    QuerySampleLibraryMulti qsl(&Inferencers);
    mlperf::TestSettings test_settings;
    mlperf::LogSettings log_settings;
    
    //mlperf와 관련된 기본 설 
    std::string scenario_str = "SingleStream";//parser.get<std::string>("--scenario");
    if (scenario_str == "Offline") {
      test_settings.scenario = mlperf::TestScenario::Offline;
    } else if (scenario_str == "Server") {
      test_settings.scenario = mlperf::TestScenario::Server;
    } else if (scenario_str == "SingleStream") {
      test_settings.scenario = mlperf::TestScenario::SingleStream;
    } else if (scenario_str == "MultiStream") {
      test_settings.scenario = mlperf::TestScenario::MultiStream;
    }
    std::string mode_str = "AccuracyOnly";//parser.get<std::string>("--mode");
    if (mode_str == "SubmissionRun") {
      test_settings.mode = mlperf::TestMode::SubmissionRun;
    } else if (mode_str == "AccuracyOnly") {
      test_settings.mode = mlperf::TestMode::AccuracyOnly;
    } else if (mode_str == "PerformanceOnly") {
      test_settings.mode = mlperf::TestMode::PerformanceOnly;
    } else if (mode_str == "FindPeakPerformance") {
      test_settings.mode = mlperf::TestMode::FindPeakPerformance;
    }

    const std::string kModelName = "AlexNet"; //"resnet50";
    #if MYSERVER
    const std::string kMlperfConfig = "/home/seoyeon/seoyeon_big/inference/multiSUT/user.conf";//parser.get<std::string>("--mlperf_config"); ##환경마다 달라짐
    #else
      const std::string kMlperfConfig = "/home/kmsguest/big-stroage/seoyeon/mlperf/inference/multiSUT/user.conf";
    #endif
    //test_settings.FromConfig(kMlperfConfig, kModelName, scenario_str);
    //test_settings.FromConfig(kUserConfig, kModelName, scenario_str);

    if (true){//parser.get<bool>("--short")) {
      test_settings.min_duration_ms = 30000;
    }

    #if MYSERVER
      log_settings.log_output.outdir = "/home/seoyeon/seoyeon_big/inference/multiSUT/output";
    #else
      log_settings.log_output.outdir = "/home/kmsguest/big-stroage/seoyeon/mlperf/inference/multiSUT/output";
    #endif
    
    //parser.get<std::string>("--output_dir");
    log_settings.log_output.prefix = "mlperf_log_";
    log_settings.log_output.suffix = "multi";//parser.get<std::string>("--suffix"); //default는 ""
    //std::cout<<log_settings.log_output.outdir.c_str()<<"\n"<<log_settings.log_output.prefix.c_str()<<"\n";
    log_settings.log_output.prefix_with_datetime = false;
    log_settings.log_output.copy_detail_to_stdout = false;
    log_settings.log_output.copy_summary_to_stdout = true;
    log_settings.log_mode = mlperf::LoggingMode::AsyncPoll;
    log_settings.log_mode_async_poll_interval_ms = 1000;
    log_settings.enable_trace = false;

    /*lg측 코드 실행*/
    mlperf::startMultiTest(&sut, &qsl, test_settings, log_settings);

    for(int i=0;i<multiN;i++)
    {
      Inferencers[i].join();
    }
    std::cout<<"done"<<std::endl;
    return 0;
  #endif
  
}