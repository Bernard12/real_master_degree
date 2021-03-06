cmake_minimum_required(VERSION 3.5)
project(master_d_tests)
set(CMAKE_CXX_STANDARD 11)
# Openmp is faster only without -O3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -mtune=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -mtune=native")

add_subdirectory(lib/googletest-release-1.8.1)
include_directories(lib/googletest-release-1.8.1/googlemock/include)
include_directories(lib/googletest-release-1.8.1/googletest/include)
include_directories(../)

add_executable(omp-test src/omp_tests.cpp)
target_link_libraries(omp-test gtest gtest_main)

add_executable(qr-test src/qr.cpp ../src/matrix/Matrix.h ../src/matrix_utils/operations.cpp ../src/matrix_utils/svd.cpp)
target_link_libraries(qr-test gtest gtest_main)

add_executable(svd-test src/svd.cpp ../src/matrix/Matrix.h ../src/matrix_utils/operations.cpp ../src/matrix_utils/svd.cpp)
target_link_libraries(svd-test gtest gtest_main)
