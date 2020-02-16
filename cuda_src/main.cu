#include "matrix/Matrix.hpp"
#include "matrix_utils/operations.cuh"
#include "matrix_utils/svd.cuh"

int main() {
    /*
    Matrix* mtr = new Matrix(5, 5);
    Matrix* dev_m;
    Matrix* dev_m2;
    Matrix* dev_m3;
    copyMatrixFromHostToDevice(mtr, &dev_m3);
    mtr->matrix[0] = 2;
    copyMatrixFromHostToDevice(mtr, &dev_m);
    copyMatrixFromHostToDevice(mtr, &dev_m2);
    // sum<<<16, 32>>>(dev_m, dev_m2);
    multiply<<<16, 32>>>(dev_m, dev_m2, dev_m3);
    multiply<<<16, 32>>>(dev_m3, .5);
    show<<<1, 1>>>(dev_m3);
    CCE(cudaGetLastError());
    CCE(cudaDeviceSynchronize())
    CCE(cudaGetLastError());
    delete mtr;
    */
    return 0;
}
