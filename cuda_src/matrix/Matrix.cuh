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
        dims = new int[2];
        dims_count = n * m;
        dims[0] = n;
        dims[1] = m;
        dims_count = 2;
        matrix = new double[n * m];
        for (int i = 0; i < n * m; i++) {
            matrix[i] = 0;
        }
    }

    Matrix(int dims_count, int* dims_value): dims_count(dims_count) {
        int total = 1;
        dims = new int[dims_count];
        for (int i = 0; i < dims_count; i++) {
            total *= dims_value[i];
            dims[i] = dims_value[i];
        }
        matrix = new double[total];
        for (int i = 0; i < total; i++) {
            matrix[i] = 0;
        }
    }

    __host__ __device__
    ~Matrix() {
        delete[] matrix;
        delete[] dims;
    }


    __host__ __device__
    double get(int i, int j) {
        if (dims_count != 2) {
            printf("Wrong sized, cur: %d, but required 2\n", dims_count);
            return 0;
        }
        // i + n * j
        int index = i + dims[0] * j;
        return matrix[index];
    }

    __host__ __device__
    void set(int i, int j, double value) {
        // i + n * j
        if (dims_count != 2) {
            printf("Wrong sized, cur: %d, but required 2\n", dims_count);
            return;
        }
        int index = i + dims[0] * j;
        matrix[index] = value;
    }

    __host__
    double get(int* indexes, int index_count) {
        if (index_count != dims_count) {
            printf("Wrong sized (cur: %d, passed: %d)\n", dims_count, index_count);
            return 0;
        }
        // for dim_count = 3
        // index for i j k
        // index = i + dims[0] * j + dims[0] * dims[1] * k;
        // int index = i;
        int index = 0;
        int mlt = 1;
        for (int i = 0; i < dims_count; i++) {
            index += indexes[i] * mlt;
            if (i != dims_count - 1) {
                mlt *= dims[i];
            }
        }
        return matrix[index];
    }

    __host__
    void set(int* indexes, int index_count, double value) {
        if (index_count != dims_count) {
            printf("Wrong sized (cur: %d, passed: %d)\n", dims_count, index_count);
            return;
        }
        // for dim_count = 3
        // index for i j k
        // index = i + dims[0] * j + dims[0] * dims[1] * k;
        // int index = i;
        int index = 0;
        int mlt = 1;
        for (int i = 0; i < dims_count; i++) {
            index += indexes[i] * mlt;
            if (i != dims_count - 1) {
                mlt *= dims[i];
            }
        }
        printf("index: %d\n", index);
        matrix[index] = value;
    }

    __host__
    void reshape(int* new_dims, int new_dims_count) {
        int total = 1;
        for (int i = 0; i < dims_count; i++) {
            total *= dims[i];
        }

        int new_total = 1;
        for (int i = 0; i < new_dims_count; i++) {
            new_total *= new_dims[i];
        }

        if (total != new_total) {
            printf("Tried to reshape matrix but got wrong shapes");
            printf("Before total: %d\n", total);
            printf("After total: %d\n", new_total);
            exit(-1);
        }


        delete[] dims;
        
        dims_count = new_dims_count;
        dims = new int[dims_count];
        for (int i = 0; i < new_dims_count; i++) {
            dims[i] = new_dims[i];
        }
    };

    __host__
    Matrix* copy() {
        Matrix* res = new Matrix(dims_count, dims);
        int total = 1;
        for (int i = 0; i < dims_count; i++) {
            total *= dims[i];
        }
        for (int i = 0; i < total; i++) {
            res->matrix[i] = matrix[i];
        }
        return res;
    }


    int n, m;
    double* matrix;
    int* dims;
    int dims_count;
};


#endif //MASTER_D_MATRIX_H
