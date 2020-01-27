cmake_minimum_required(VERSION 3.5)
project(master_d)

set(CMAKE_CXX_STANDARD 11)

add_executable(master_d
        src/main.cpp
        src/matrix/Matrix.cpp
        src/matrix/Matrix.h
        src/matrix_utils/operations.cpp
        src/matrix_utils/operations.h
        src/matrix_utils/svd.h
        src/matrix_utils/svd.cpp
        )

add_subdirectory(tests)