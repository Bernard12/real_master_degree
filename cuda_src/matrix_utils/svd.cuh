//
// Created by ivan on 11.01.2020.
//

#ifndef MASTER_D_SVD_H
#define MASTER_D_SVD_H
#include "../matrix/Matrix.cuh"
#include "operations.cuh"

struct Triple {
    Triple(Matrix* a, Matrix* b, Matrix* c): first(a), second(b), third(c) {}
    Matrix* first, *second, *third;
};


pair<Matrix*, Matrix*> QRDecompositionNaive(Matrix* a);
Triple* SVDDecomposition(Matrix *a, int rank, double eps);
Triple* SVDDecompositionwCUB(Matrix *a, int rank, double eps);
Triple* SVDDecompositionNaive(Matrix* a);

void copyMatrixFromHostToDevice(Matrix* hostMatrix, Matrix** deviceMatrix, double** deviceMatrixArray, int** deviceDimsArray);
#endif //MASTER_D_SVD_H
