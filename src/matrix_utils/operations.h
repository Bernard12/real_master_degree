//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_OPERATIONS_H
#define MASTER_D_OPERATIONS_H

#include "../matrix/Matrix.h"
#include <stdexcept>
#include <random>
#include <chrono>

Matrix sum(Matrix &a, Matrix &b);

Matrix multiply(Matrix &a, Matrix &b);

Matrix multiply(Matrix &a, double &b);

bool equals(Matrix &a, Matrix &b, double eps);

double vectorColLength(Matrix& a);

Matrix randomMatrix(int n, int m);

Matrix vectorColNormalize(Matrix &a);

Matrix transpose(Matrix& a);

Matrix subMatrix(Matrix& a, int rowStart, int rowEnd, int colStart, int colEnd);

#endif //MASTER_D_OPERATIONS_H
