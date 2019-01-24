cmake_minimum_required(VERSION 3.13)
project(seminar)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 -lOpenCL -lfreeimageplus")

add_executable(seminar util.cpp serial.cpp main.cpp)

find_package(OpenCL REQUIRED)
target_link_libraries(seminar OpenCL::OpenCL)

find_path(SIFTGPU_INCLUDE_DIR FreeImagePlus.h)
find_library(SIFTGPU_LIBRARY libfreeimageplus)
target_link_libraries(seminar freeimageplus)