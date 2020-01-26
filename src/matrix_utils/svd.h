//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_SVD_H
#define MASTER_D_SVD_H
#include "../matrix/Matrix.h"
#include "operations.h"
double getBiggestEugenValueOfSquareMatrix(Matrix &a);
double getBiggestSingularValueOfSquareMatrix(Matrix &a);
pair<Matrix, Matrix> QRDecompositionNaive(Matrix& a);
#endif //MASTER_D_SVD_H
