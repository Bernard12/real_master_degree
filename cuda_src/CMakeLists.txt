cmake_minimum_required(VERSION 3.5)
project(master_d_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 --expt-relaxed-constexpr -arch=sm_60")

find_package(CUDA)

add_executable(cd
  main.cu
  matrix/Matrix.cuh
  matrix_utils/operations.cu
  matrix_utils/svd.cu
)

target_link_libraries(cd -lcublas -lcusolver)