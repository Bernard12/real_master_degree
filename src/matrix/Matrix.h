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
        this->matrix = vector<vector<double>>(n, vector<double>(m, 0));
    }

    vector<double> &operator[](const int val) {
        return (this->matrix)[val];
    }

    pair<int, int> shape() {
        return make_pair(this->n, this->m);
    }

    void show() {
        cout << "-----MATRIX(" << n << "," << m << ")------\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                cout << (this->matrix)[i][j] << " ";
            }
            cout << '\n';
        }
        cout << "-----------\n";
    }

private:
    int n, m;
    vector<vector<double> > matrix;
};


#endif //MASTER_D_MATRIX_H
