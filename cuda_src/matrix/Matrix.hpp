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
    Matrix(int n, int m) : n(n), m(m) {
        matrix = new double[n * m];
        for (int i = 0; i < n * m; i++) {
            matrix[i] = 0;
        }
    }

    ~Matrix() {
        delete[] matrix;
    }


    __device__ __host__
    double get(int i, int j) {
        int index = m * i + j;
        return matrix[index];
    }

    __device__ __host__
    void set(int i, int j, double value) {
        int index = m * i + j;
        matrix[index] = value;
    }

    int n, m;
    double* matrix;
};


#endif //MASTER_D_MATRIX_H
