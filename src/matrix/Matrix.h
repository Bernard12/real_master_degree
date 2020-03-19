//
// Created by ivan on 08.01.2020.
//

#ifndef MASTER_D_MATRIX_H
#define MASTER_D_MATRIX_H

#include <iostream>
#include <vector>

using namespace std;

class Matrix {

public:
    Matrix(int n, int m) : n(n), m(m) {
        total_element_count = n * m;
        matrix = new double[total_element_count];
        for (int i = 0; i < total_element_count; i++) {
            matrix[i] = 0;
        }
        real_shape = new int[2];
        real_shape[0] = n;
        real_shape[1] = m;
        shape_length = 2;
    }

    ~Matrix() {
        delete[] matrix;
        delete[] real_shape;
    }

    pair<int, int> shape() {
        if (shape_length != 2) {
            printf("Deprecated method for shape when real shape is : %d\n", shape_length);
            exit(12);
        }
        return make_pair(this->n, this->m);
    }

    double get(int i, int j) {
        int index = i + n * j;
        return matrix[index];
    }

    void set(int i, int j, double value) {
        int index = i + n * j;
        matrix[index] = value;
    }

    void show() {
        cout << "-----MATRIX(" << n << "," << m << ")------\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int index = i + n * j;
                cout << matrix[index] << " ";
            }
            cout << '\n';
        }
        cout << "-----------\n";
    }

    int n, m;
    double *matrix;

    int shape_length;
    int *real_shape;
    int total_element_count;
};


#endif //MASTER_D_MATRIX_H
