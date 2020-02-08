cmake_minimum_required(VERSION 3.5)
project(master_d_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA)

add_executable(cd
  main.cu
  matrix/Matrix.hpp
  )
