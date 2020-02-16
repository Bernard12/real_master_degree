cmake_minimum_required(VERSION 3.5)
project(master_d_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

find_package(CUDA)

add_executable(cd
  main.cu
  matrix/Matrix.hpp
  matrix_utils/operations.cu
  matrix_utils/svd.cu
  )
