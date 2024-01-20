cmake_minimum_required(VERSION 3.27)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(task3 CUDA CXX)

#find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(./utils ${PNG_INCLUDE_DIR})

set(SOURCES main.cu ./utils/pngio.cpp)

set(CMAKE_CUDA_STANDARD 17)
list(APPEND CMAKE_CXX_FLAGS "-O2 -ffast-math -Wall ${OpenMP_CXX_FLAGS}")
#list(APPEND CUDA_NVCC_FLAGS "-g -G")

add_executable(task3 ${SOURCES})

target_link_libraries(task3 /usr/local/cuda/lib64/libcudart.so)
target_link_libraries(task3 ${PNG_LIBRARY})

set_target_properties(task3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
