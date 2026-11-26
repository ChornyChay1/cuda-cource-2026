cmake_minimum_required(VERSION 3.18)
project(lab2 LANGUAGES CXX CUDA)

add_executable(submission src/main.cu src/matmul.cu)
set_target_properties(submission PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)

option(LAB2_BUILD_BENCHMARK "Build the cuBLAS and shared memory benchmark" OFF)
if(LAB2_BUILD_BENCHMARK)
  find_package(CUDAToolkit REQUIRED)
  add_executable(benchmark src/benchmark.cu src/matmul.cu)
  set_target_properties(benchmark PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
  target_link_libraries(benchmark PRIVATE CUDA::cublas)
endif()
