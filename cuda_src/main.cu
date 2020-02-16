#include "matrix/Matrix.cuh"
#include "matrix_utils/svd.cuh"
#include <stdio.h>

#define CCE(errValue)                                        \
    do {                                                                \
        if (errValue != cudaSuccess) {                                  \
            fprintf(stderr ,"[CUDA-ERROR]-[%s(line:%d)]: %s\n", __FILE__, __LINE__, cudaGetErrorString(errValue)); \
            exit(0);                                                    \
        }                                                               \
    } while(0);

// CATCH_CUDA_ERR(cudaMalloc(&dev_array, sizeof(int) * used_n));
// CATCH_CUDA_ERR(cudaMemcpy(dev_array, array, sizeof(int) * used_n, cudaMemcpyHostToDevice));

void copyMatrixFromHostToDevice(Matrix* hostMatrix, Matrix** deviceMatrix) {
    const int n = hostMatrix->n, m = hostMatrix->m;
    Matrix* temp = new Matrix(n, m);

    const int matrix_size = sizeof(double) * n * m;
    CCE(cudaMalloc(&temp->matrix, matrix_size));
    CCE(cudaMemcpy(temp->matrix, hostMatrix->matrix, sizeof(Matrix) * 1, cudaMemcpyHostToDevice));

    CCE(cudaMalloc(deviceMatrix, sizeof(Matrix) * 1));
    CCE(cudaMemcpy(*deviceMatrix, temp, sizeof(Matrix) * 1, cudaMemcpyHostToDevice));

    temp-> matrix = NULL;
    delete temp;

}

int main() {
    Matrix* mtr = hilbert(1000, 1000);
    // auto p = QRDecompositionNaive(mtr);
    // Matrix* res = multiply(p.first, p.second);
    // show(mtr, 5, 5);
    // show(res, 5, 5);
    auto trip = SVDDecomposition(mtr, 10, 1e-6);
    auto tmp = multiply(trip->first, trip->second);
    auto t = transpose(trip->third);
    auto res = multiply(tmp, t);
    // printf("Diff: %f \n", diff(mtr, res));
    printf("Diff: %f \n", diff(mtr, res));
    delete mtr;
    return 0;
}
