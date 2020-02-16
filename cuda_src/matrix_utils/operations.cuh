//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_OPERATIONS_H
#define MASTER_D_OPERATIONS_H

#include "../matrix/Matrix.hpp"
#include <random>
#include <chrono>


__global__
void get(Matrix* mtr, int index, double* res);

__global__
void set(Matrix* mtr, int index, double* res);

__global__
void show(Matrix* mtr);

__global__
void sum(Matrix* a, Matrix* b);

__global__
void multiply(Matrix* a, Matrix* b, Matrix* res);

__global__
void multiply(Matrix* a, double* b);

__global__
void equals(Matrix* a, Matrix* b, double eps, int* res);

void diff(Matrix* a, Matrix* b, double* diff);

void randomMatrix(int n, int m, Matrix* res);

void vectorColLength(Matrix* a, double* res);

void matrixNorm(Matrix* a, double* res);

void vectorColNormalize(Matrix* a);

__host__ __device__
void transpose(Matrix* a, Matrix* res);

__host__ __device__
void subMatrix(Matrix* a, int rowStart, int rowEnd, int colStart, int colEnd, Matrix* res);

__global__
void hilbert(int n, int m, Matrix** res);
#endif //MASTER_D_OPERATIONS_H
