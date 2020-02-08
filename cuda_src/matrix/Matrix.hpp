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


    pair<int, int> shape() {
        return make_pair(this->n, this->m);
    }

    __device__
    double get(int i, int j) {
        int index = m * i + j;
        return matrix[index];
    }

    void set(int i, int j, double value) {
        int index = m * i + j;
        matrix[index] = value;
    }


    int n, m;
    double* matrix;
};

__global__
void show(Matrix* mtr, int n, int m) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    printf("-----MATRIX(%d, %d)-------\n", n, m);
    printf("StartIdx(%d), offset(%d)", startIdx, offset);
    for (int ij = startIdx; ij < n * m; ij+= offset) {
        int i = ij / m, j = ij % m;

        if (j == 0) {
            printf("\n");
        }

        printf("%f ", mtr->get(i, j));
    }
    printf("\n-----------\n");
}


#endif //MASTER_D_MATRIX_H
