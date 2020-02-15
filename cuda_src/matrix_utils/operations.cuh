//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_OPERATIONS_H
#define MASTER_D_OPERATIONS_H

#include "../matrix/Matrix.hpp"
#include <random>

__global__
void show(Matrix* mtr);

__global__
void sum(Matrix* a, Matrix* b);

__global__
void multiply(Matrix* a, Matrix* b, Matrix* res);

__global__
void multiply(Matrix* a, double b);

__global__
void equals(Matrix* a, Matrix* b, double eps, int* res);

/*
__global__
double diff(Matrix* a, Matrix* b);

__global__
double vectorColLength(Matrix* a);

__global__
double matrixNorm(Matrix* a);

__global__
Matrix* randomMatrix(int n, int m);

__global__
Matrix* vectorColNormalize(Matrix* a);

__global__
Matrix* transpose(Matrix* a);

__global__
Matrix* subMatrix(Matrix* a, int rowStart, int rowEnd, int colStart, int colEnd);

__global__
Matrix* hilbert(int n, int m);
*/
#endif //MASTER_D_OPERATIONS_H
