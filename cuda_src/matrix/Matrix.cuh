//
// Created by ivan on 08.01.2020.
//

#ifndef MASTER_D_MATRIX_H
#define MASTER_D_MATRIX_H

#include <stdio.h>
#include <vector>

using namespace std;

class Matrix {

public:

    __host__ __device__
    Matrix(int n, int m) : n(n), m(m), dims_count(n * m) {
        dims = new int[2];
        dims[0] = n;
        dims[1] = m;
        matrix = new double[n * m];
        for (int i = 0; i < n * m; i++) {
            matrix[i] = 0;
        }
    }

    Matrix(int dims_count, int* dims_value, int fake) {
        int total = 1;
        for (int i = 0; i < dims_count; i++) {
            total *= dims_value[i];
        }
        matrix = new double[total];
        dims = new int[total];
        for (int i = 0; i < total; i++) {
            dims[i] = 0;
        }
    }

    __host__ __device__
    ~Matrix() {
        delete[] matrix;
        delete[] dims;
    }


    __host__ __device__
    double get(int i, int j) {
        int index = i + dims[0] * j;
        return matrix[index];
    }

    __host__ __device__
    void set(int i, int j, double value) {
        int index = i + dims[0] * j;
        matrix[index] = value;
    }

    int n, m;
    double* matrix;
    int* dims;
    int dims_count;
};


#endif //MASTER_D_MATRIX_H
