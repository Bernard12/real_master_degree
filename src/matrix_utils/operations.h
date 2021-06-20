//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_OPERATIONS_H
#define MASTER_D_OPERATIONS_H

#include "../matrix/Matrix.h"
#include <stdexcept>
#include <random>
#include <chrono>

Matrix* sum(Matrix* a, Matrix* b);

Matrix* multiply(Matrix* a, Matrix* b);

Matrix* multiply(Matrix* a, double b);

bool equals(Matrix* a, Matrix* b, double eps);

double diff(Matrix* a, Matrix* b);

double vectorColLength(Matrix* a);

double matrixNorm(Matrix* a);

double frobeniousMatrixNorm(Matrix* a);

Matrix* randomMatrix(int n, int m);

Matrix* vectorColNormalize(Matrix* a);

Matrix* transpose(Matrix* a);

Matrix* subMatrix(Matrix* a, int rowStart, int rowEnd, int colStart, int colEnd);

Matrix* hilbert(int n, int m);

Matrix* hilbert(int n, int m, int k);
Matrix* hilbert(int i1, int i2, int i3, int i4, int i5);
Matrix* hilbert(int i1, int i2, int i3, int i4, int i5, int i6, int i7);

Matrix* sinCube(int r, double step);
double convolution(vector<Matrix*> tt, vector<Matrix *> u);
#endif //MASTER_D_OPERATIONS_H
