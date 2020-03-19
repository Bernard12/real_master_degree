//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_SVD_H
#define MASTER_D_SVD_H

#include <lapacke.h>
#include "../matrix/Matrix.h"
#include "operations.h"

struct Triple {
    Triple(Matrix* a, Matrix* b, Matrix* c): first(a), second(b), third(c) {}
    Matrix* first, *second, *third;
};
double getBiggestEugenValueOfSquareMatrix(Matrix *a);
double getBiggestSingularValueOfSquareMatrix(Matrix *a);
pair<Matrix*, Matrix*> QRDecompositionNaive(Matrix* a);
Triple* SVDDecomposition(Matrix *a);
Triple* SVDDecompositionNaive(Matrix* a, int rank, double eps);
#endif //MASTER_D_SVD_H
