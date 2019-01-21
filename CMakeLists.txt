#cmake_minimum_required(VERSION 3.13)
#project(seminar)

#set(CMAKE_CXX_STANDARD 11)

# opencl
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -I/opt/cuda/include -lOpenCL")

# freeimage
#find_path(SIFTGPU_INCLUDE_DIR FreeImage.h)
#find_library(SIFTGPU_LIBRARY libfreeimageplus)
#include_directories(${SIFTGPU_INCLUDE_DIR})

#add_executable(seminar util.cpp serial.cpp main.cpp)
#target_link_libraries(seminar freeimageplus)


cmake_minimum_required(VERSION 3.12)
project(seminar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -lOpenCL -lfreeimageplus -ffast-math")

add_executable(seminar util.cpp serial.cpp main.cpp)


find_package(OpenCL REQUIRED)
target_link_libraries(seminar OpenCL::OpenCL)

find_path(SIFTGPU_INCLUDE_DIR FreeImagePlus.h)
find_library(SIFTGPU_LIBRARY libfreeimageplus)
target_link_libraries(seminar freeimageplus)