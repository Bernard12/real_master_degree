cmake_minimum_required(VERSION 3.5)
project(master_d_tests)
set(CMAKE_CXX_STANDARD 11)

add_subdirectory(lib/googletest-release-1.8.1)
include_directories(lib/googletest-release-1.8.1/googlemock/include)
include_directories(lib/googletest-release-1.8.1/googletest/include)
include_directories(../)

add_executable(simple-test src/simple_test.cpp)
target_link_libraries(simple-test gtest gtest_main)

add_executable(qr-test src/qr.cpp ../src/matrix/Matrix.h ../src/matrix_utils/operations.cpp ../src/matrix_utils/svd.cpp)

target_link_libraries(qr-test gtest gtest_main)
