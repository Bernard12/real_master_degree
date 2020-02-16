//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_OPERATIONS_H
#define MASTER_D_OPERATIONS_H

#include "../matrix/Matrix.cuh"
#include <random>
#include <chrono>

__host__ __device__
void show(Matrix* mtr, int n, int m);

__host__ __device__
Matrix* sum(Matrix* a, Matrix* b);

__host__ __device__
Matrix* multiply(Matrix* a, Matrix* b);

__host__ __device__
Matrix* multiply(Matrix* a, double b);

__host__ __device__
bool equals(Matrix* a, Matrix* b, double eps);

__host__ __device__
double diff(Matrix* a, Matrix* b);

__host__ __device__
double vectorColLength(Matrix* a);

__host__ __device__
double matrixNorm(Matrix* a);

__host__
Matrix* randomMatrix(int n, int m);

__host__ __device__
Matrix* vectorColNormalize(Matrix* a);

__host__ __device__
Matrix* transpose(Matrix* a);

__host__ __device__
Matrix* subMatrix(Matrix* a, int rowStart, int rowEnd, int colStart, int colEnd);

__host__ __device__
Matrix* hilbert(int n, int m);
#endif //MASTER_D_OPERATIONS_H