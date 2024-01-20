cmake_minimum_required(VERSION 3.27)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(task2 CUDA)

set(CMAKE_CUDA_STANDARD 17)
list(APPEND CUDA_NVCC_FLAGS "-g -G")

add_executable(task2 main.cu)

set_target_properties(task2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
