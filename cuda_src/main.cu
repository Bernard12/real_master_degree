#include "matrix/Matrix.hpp"
#include "matrix_utils/svd.hpp"

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
    auto host_shape = hostMatrix->shape();
    const int n = host_shape.first, m = host_shape.second;
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
    Matrix* dev_m;
    copyMatrixFromHostToDevice(mtr, &dev_m);
    delete mtr;
    show<<<1, 1>>>(dev_m, 5, 5);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize())
    CCE(cudaGetLastError());
    return 0;
}
