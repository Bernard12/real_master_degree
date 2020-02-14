#include "matrix/Matrix.hpp"
#include "matrix_utils/operations.cuh"

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
    Matrix* mtr = new Matrix(5, 5);
    mtr->matrix[0] = 1;
    Matrix* dev_m;
    Matrix* dev_m2;
    copyMatrixFromHostToDevice(mtr, &dev_m);
    copyMatrixFromHostToDevice(mtr, &dev_m2);
    sum<<<16, 32>>>(dev_m, dev_m2);
    show<<<1, 1>>>(dev_m, 5, 5);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize())
    CCE(cudaGetLastError());
    delete mtr;
    return 0;
}
