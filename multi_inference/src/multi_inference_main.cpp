/*벤치마킹을 실행시키는 main파일*/
//인용한 코드는 [인용]이라고 표시함
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

//./include 폴더의 헤더함수들
#include "Inferencer.h"

std::vector<at::cuda::CUDAStream> streams;

/*[인용]query_send_thread를 제어하기 위한 bsem*/
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
//LoadGen과 Inferencer들이 접근하는 큐
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
    /*응답 push: 새 응답이 들어오면 query_send_thread 대기 종료 신호 전송*/
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
    void destroy() {
        front = NULL;
        rear = NULL;
        len = 0;
        bsem_reset(has_jobs);
        free(has_jobs);
    }
};
/*쿼리 큐: query_send_thread로부터 받은 쿼리를 담으며 Inferencer들이 하나씩 가지고 있음*/
//[인용]사피온(Sapeon) 측에서 작성한 BufferQueue를 카피하여 사용함
template <typename T>
class QueryQueue {
 public:
  QueryQueue(const size_t max_items)
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
/*val_map.txt파일의 문장 분리를 위한 함수*/
//[인용]사피온(Sapeon) 측에서 작성한 BufferQueue를 카피하여 사용함
std::vector<std::string> SplitE1(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;
  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }
  return answer;
};

/*한 모델에 대한 추론을 하는 Inferencer 클래스*/
class Inferencer
{
private:
    torch::jit::script::Module m;
    pthread_t thread_;
    int id;  
    QueryQueue<mlperf::QuerySample>* query_queue = nullptr; 
    std::vector<std::string> total_input_filenames_;  /*데이터 셋의 파일 이름 저장 벡터*/
    std::vector<int> total_expected_labels_;          /*데이터 셋의 라벨(정답) 저장 벡터*/
    const size_t kMaxBatch = 18;                      /*분석 필요*/
    const size_t image_size_ = 224 * 224 * 3;         
public:
    std::vector<cv::Mat> images;                      /*데이터셋의 샘플 이미지 저장 벡터*/
    responseQueue *response_queue;                    /*Inferencer들이 공유하는 응답큐*/
    std::string dataset_path_;                        /*데이터셋 폴더 경로*/
    std::string model_name_;
    size_t total_sample_count_;
    size_t performance_sample_count_;
    mlperf::TestMode mlperf_mode_;
    int result_good=0;                               /*모델 정확도 판별 변수*/
    int result_total=0;
    std::vector<double> result_timing;               /*추론 시간들을 저장하는 벡터*/
    std::chrono::time_point<std::chrono::high_resolution_clock, std::chrono::nanoseconds> last_loaded; /*데이터셋에 접근한 최종시간*/
    bool inferencers_keepalive;                      /*추론 지속 여부에 대한 변수*/

    Inferencer(){inferencers_keepalive=true; };
    /*클래스의 응답큐 초기화. SUT과 LoadGen이 공유하는 큐이기 때문에 포인터로 가지고 있음*/
    void init_responseQ(responseQueue* rq)
    {
      response_queue=rq;
    }
    /*클래스 초기화 함수. model load 및 dataset내의 파일이름 저장, 클래스 초기화 */
    int Init(const char* model_path,char*& dataset_path,const char* model_name,int id) 
    {
      this->id=id;
      query_queue = new QueryQueue<mlperf::QuerySample>(kMaxBatch);
      dataset_path_=dataset_path;
      std::ifstream openFile("/home/seoyeon/seoyeon_big/inference/multiSUT/fake_imagenet/val_map.txt");//이미지 정보 및 추론 정답이 저장된 txt 파일
      if (openFile.is_open()) {
          std::string line;
          while (getline(openFile, line)) {
            std::vector<std::string> result = SplitE1(line, ' '); /*valmap파일을 한줄씩 읽어서 분리 ex) val/800px-Porsche_991_silver_IAA.jpg 817*/
            std::string filename = dataset_path_+"/"+result[0];
            total_expected_labels_.push_back(stoi(result[1]));
            total_input_filenames_.push_back(filename);
          }
          openFile.close();
      }
      total_sample_count_ = total_input_filenames_.size();
      performance_sample_count_ = total_sample_count_;
      model_name_ = model_name;
      /*모델 로드 및 gpu에 올려두기, 모델 작동 테스트*/
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
    static void* worker_thread(Inferencer *inferencer)
    {
      at::Device device(torch::kCUDA);
      while(inferencer->inferencers_keepalive) 
      {
        auto buffered_inputs = inferencer->query_queue->Pop();
        if(buffered_inputs.size()>0){ 
          int max_idx;
          struct mlperf::QuerySampleResponse response_log;
          struct mlperf::QuerySampleResponse *response = new mlperf::QuerySampleResponse;
          auto start = std::chrono::high_resolution_clock::now(); 
          /*추론*/
          const mlperf::QuerySample& query_sample = buffered_inputs[0];
          inputs.clear();
          at::Tensor input_tensor = torch::from_blob(inferencer->images[query_sample.index].data, {1, 224,224,3},torch::kFloat32);
          input_tensor = input_tensor.permute({0,3,1,2}); 
          input_tensor = input_tensor.to(device,torch::kFloat32);
          at::Tensor result = inferencer->m.forward({input_tensor}).toTensor();
          result = result.to(torch::kCPU);
          /*모델정확도 확인*/
          auto max_result = torch::argmax(result,1);
          max_idx = max_result.item<int>();
          if(max_idx==inferencer->total_expected_labels_[query_sample.index])
            inferencer->result_good++;
          inferencer->result_total++;
          /*후처리*/
          auto later = std::chrono::high_resolution_clock::now();
          auto duration = std::chrono::duration<double, std::milli>(later - start);
          inferencer->result_timing.push_back(duration.count()*0.001);
          std::vector<std::array<uint8_t, sizeof(float)>> post_results(1);
          uintptr_t bi;
          std::memcpy(post_results[0].data(), &max_idx, sizeof(int)); 
          bi = reinterpret_cast<uintptr_t>(post_results[0].data());
          /*응답 데이터 생성*/
          response_log.id = buffered_inputs[0].id;
          response_log.data = bi;
          response_log.size = 4;
          response_log.DNNindex=buffered_inputs[0].DNNindex;
          response->id = buffered_inputs[0].id;
          response->data = bi;
          response->size = 4;
          response->DNNindex=buffered_inputs[0].DNNindex;
          //응답 전송&로그저장
          inferencer->response_queue->push(response);
          QuerySamplesResponseLogging(&response_log,buffered_inputs.size());
          }
        }
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(now - inferencer->last_loaded);
        inferencer->add_results("TestScenario.SingleStream",inferencer->model_name_
                                ,inferencer->result_timing,duration.count()*0.001,true);
        return NULL;
    };
    void join()
    {
        pthread_join(thread_,NULL);  //join의 사용법이 잘못되었을 수도 있음 체크
    }
    //worker_thread함수를 스레드로 동작하도록
    void start()  
    {
        pthread_create(&thread_,NULL,(void*(*)(void*))worker_thread,this);
    }
    /*SUTMulti로부터 쿼리 수신*/
    void SutIssueQuery(const std::vector<mlperf::QuerySample>& queries)  //원래이름 samples
    {
      //한개의 쿼리만 전송되었으면 한번만 push
      for (auto query : queries) {  
        query_queue->Push(query);
      }
    };
    void SutFlushQueries(){};
    /*이미지 전처리 함수들: resize_with_aspectratio,center_crop,pre_process_img
      MLPerf-Dataset의 전처리와 동일하게 동작하도록 같은 방식으로 전처리함*/
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
    cv::Mat pre_process_img(cv::Mat img) 
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
    void QslLoadSamplesToRam(){
        for (const auto& filename : total_input_filenames_) {
            cv::Mat img = cv::imread(filename);
            cv::Mat pre_proc_img;
            pre_proc_img = pre_process_img(img);//이미지 전처리 수행
            images.push_back(pre_proc_img);
        }
        last_loaded = std::chrono::high_resolution_clock::now();
        std::cout<<"load done\n";
    };
    void QslUnloadSamplesFromRam(
        const std::vector<mlperf::QuerySampleIndex>& samples){
        //로드한 데이터셋을 메모리해제하는 작업 추가해야함
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

class SUTMulti : public mlperf::SystemUnderTest {
 private:
  Inferencer(* inferencers)[2];
  //모델 클래스  
  std::string name_{"MultiSUT"};
  responseQueue * responseQ;
 public:
  SystemUnderTestMulti(Inferencer(* i)[2])
  : inferencers(i)/*멤버 이니셜 라이즈*/ { 
    responseQ=new responseQueue();
    //(*inferencers)[0] == 1번째 inferencer 클래스
    (*inferencers)[0].init_responseQ(responseQ); 
    (*inferencers)[1].init_responseQ(responseQ);
  }
  ~SystemUnderTestMulti() override = default;
  const std::string& Name() override { return name_; }
  /*query_send_thread 해당 함수 호출. 맞는 inferencer에 쿼리 보냄*/
  void IssueQuery(const std::vector<mlperf::QuerySample>& queries) 
  override {
    /*query_send_thread의 대기를 바로 종료시키기 위한 장치 */
    mlperf::QueryReceiveComplete(queries[0].id); 
    if(queries[0].DNNindex==0)
      (*inferencers)[0].SutIssueQuery(queries);
    else
      (*inferencers)[1].SutIssueQuery(queries);
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
  void inferencer_end(int id)
    {
      (*inferencers)[id].inferencers_keepalive=false;
    };
};

class QSLMulti : public mlperf::QuerySampleLibrary {
 private:
  Inferencer(* inferencers)[2]; //배열 포인터.                         배열의 주소를 가지고있음. 포인터 배열과 다름 [2]가리키는 포인터라는 의미
  std::string name_{"MultiQSL"};

 public:
  QuerySampleLibraryMulti(Inferencer(* i)[2]) : inferencers(i){  }
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
    (*inferencers)[0].QslLoadSamplesToRam();
    (*inferencers)[1].QslLoadSamplesToRam();
  }
  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex>& samples) override {
    (*inferencers)[0].QslUnloadSamplesFromRam(samples);
  }
};


//stream 테스트: 2개의 stream을 만들어 streams에 저장해둠
void streamTest(){
  std::cout<<"streamTest()\n";
  for(int i=0; i<2; i++){
    streams.push_back(at::cuda::getStreamFromPool(true,0));
  }
}

int main(int argc,char *argv[])
{
  int multiN=2;
  class Inferencer Inferencers[2];                            /*inferencer 들 초기화(동적 문제로 인해 현재는 고정 크기의 배열로 설정함)*/
  for(int i=0;i<multiN;i++)
  {
      if(i==0)
        Inferencers[i].Init(argv[1],argv[2],argv[3],i);
      else 
        Inferencers[i].Init(argv[1],argv[2],argv[3],i);
        //Init(모델 절대경로,데이터셋 절대경로,모델 이름, id)
      Inferencers[i].start();
  }
  SystemUnderTestMulti sut(&Inferencers);
  QuerySampleLibraryMulti qsl(&Inferencers);
  mlperf::TestSettings test_settings;
  mlperf::LogSettings log_settings;
  
  //mlperf와 관련된 기본 설정
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
  const std::string kModelName = "resnet50";
  const std::string kMlperfConfig = "/home/seoyeon/seoyeon_big/inference/multiSUT/user.conf";

  if (true){//parser.get<bool>("--short")) {
    test_settings.min_duration_ms = 30000;
  }
    log_settings.log_output.outdir = "/home/seoyeon/seoyeon_big/inference/multiSUT/output";
  
  log_settings.log_output.prefix = "mlperf_log_";
  log_settings.log_output.suffix = "multi";
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
  return 0;
}