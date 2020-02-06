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

    double get(int i, int j) {
        int index = m * i + j;
        return matrix[index];
    }

    void set(int i, int j, double value) {
        int index = m * i + j;
        matrix[index] = value;
    }

    void show() {
        cout << "-----MATRIX(" << n << "," << m << ")------\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int index = i * m + j;
                cout << matrix[index] << " ";
            }
            cout << '\n';
        }
        cout << "-----------\n";
    }

private:
    int n, m;
    double* matrix;
};


#endif //MASTER_D_MATRIX_H
