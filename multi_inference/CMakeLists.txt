#cmake 프로그램의 최소 버전
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#프로젝트 정보
project(Multi-Mlperf)

message(STATUS "CMAKE_SOURCE_DIR : ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR : ${CMAKE_CURRENT_SOURCE_DIR}") 

#set으로 이름, 경로등 지정
set(TARGET_NAME multi_mlperf) #실행파일 이름 정의.
set(LOADGEN_DIR ../loadgen)         #/inference/loadgen
set(HEADER_DIR ./include)           #/inference/multi_inference/include
SET(SRC_FILES src/multi_inference_main.cpp src/A.cpp src/Inferencer.cpp)

#loadgen폴더에 있는 CMakeList를 이용해서 빌드하기 위함.
add_subdirectory(${LOADGEN_DIR} mlperf_loadgen) 

# undefined reference to symbol 'pthread_join@@GLIBC_2.2.5' 이 오류가 발생할때 아래 두 코드 사용할 것,
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(Torch REQUIRED) #cmake의 라이브러리 검색 모듈
find_package(OpenCV REQUIRED)
#include_directories(${})

#opencv를 쓰려고했지만 오류있음: undefined reference to `cv::imread(std::string const&, int)'
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#link_directories( ${OpenCV_LIB_DIR} )

#타겟: 실행파일, 라이브러리, 사용자 명령어
add_executable(${TARGET_NAME} ${SRC_FILES}) #타겟을 만들고, 타겟에 필요한 소스파일 나열. 

#include_directories(${OPENCV_DIR}/include)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR} ${LOADGEN_DIR} ${HEADER_DIR}) #타겟이 헤더파일을 찾기 위해서
target_link_libraries(${TARGET_NAME} mlperf_loadgen ${TORCH_LIBRARIES} ${OpenCV_LIBS}) #빌드에 필요한 타겟을 나열/실행파일, 라이브러리, 외부프로젝트
#set_property(TARGET multi PROPERTY CXX_STANDARD 14)