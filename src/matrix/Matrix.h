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
    Matrix(int n, int m) {
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
        return make_pair(this->n(), this->m());
    }

    void reshape(vector<int> new_shapes) {
        int new_total = 1;
        for (int new_shape : new_shapes) {
            new_total *= new_shape;
        }

        if (new_total != total_element_count) {
            printf("Cannot reshape, new_total before:%d, new_total after: %d\n", total_element_count, new_total);
            exit(-1);
        }

        delete[] real_shape;

        shape_length = new_shapes.size();
        real_shape = new int[new_shapes.size()];
        for (int i = 0; i < new_shapes.size(); i++) {
            real_shape[i] = new_shapes[i];
        }
    }

    Matrix *copy() {
        Matrix *res = new Matrix(0, 0);

        res->shape_length = shape_length;
        res->total_element_count = total_element_count;

        res->matrix = new double[total_element_count];
        for (int i = 0; i < total_element_count; ++i) {
            res->matrix[i] = matrix[i];
        }
        res->real_shape = new int[shape_length];
        for (int j = 0; j < shape_length; ++j) {
            res->real_shape[j] = real_shape[j];
        }
        return res;
    }

    double get(int i, int j) {
        int index = i + real_shape[0] * j;
        return matrix[index];
    }

    double get(int i, int j, int k) {
        if (shape_length != 3) {
            printf("Matrix is not 3d cannot get element");
            exit(13);
        }
        int index = i + real_shape[0] * j + real_shape[0] * real_shape[1] * k;
        return matrix[index];
    }


    Matrix *get2DshapeFrom3d(int x) {
        if (shape_length != 3) {
            printf("Cannot get shape from non 3d matrix");
            exit(14);
        }
        Matrix *res = new Matrix(real_shape[0], real_shape[2]);
        for (int i = 0; i < real_shape[1]; i++) {
            for (int j = 0; j < real_shape[2]; j++) {
                res->set(i, j, this->get(x, i, j));
            }
        }
        return res;
    }

    void set(int i, int j, double value) {
        int index = i + real_shape[0] * j;
        matrix[index] = value;
    }

    void show() {
        int n = this->n();
        int m = this->m();
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

    int n() {
        if (shape_length != 2) {
            printf("Current shape is not 2 (cur %d)\n", shape_length);
            exit(12);
        }
        return real_shape[0];
    }

    int m() {
        if (shape_length != 2) {
            printf("Current shape is not 2 (cur %d)\n", shape_length);
            exit(12);
        }
        return real_shape[1];
    }

    double *matrix;

    int shape_length;
    int *real_shape;
    int total_element_count;
};


#endif //MASTER_D_MATRIX_H
