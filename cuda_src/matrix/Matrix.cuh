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
    Matrix(int n, int m) : n(n), m(m) {
        matrix = new double[n * m];
        for (int i = 0; i < n * m; i++) {
            matrix[i] = 0;
        }
    }

    __host__ __device__
    ~Matrix() {
        delete[] matrix;
    }


    __host__ __device__
    double get(int i, int j) {
        int index = i + n * j;
        return matrix[index];
    }

    __host__ __device__
    void set(int i, int j, double value) {
        int index = i + n * j;
        matrix[index] = value;
    }

    int n, m;
    double* matrix;
};


#endif //MASTER_D_MATRIX_H
