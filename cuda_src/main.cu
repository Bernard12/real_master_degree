#include "matrix/Matrix.cuh"
#include "matrix_utils/svd.cuh"
#include <stdio.h>

// CATCH_CUDA_ERR(cudaMalloc(&dev_array, sizeof(int) * used_n));
// CATCH_CUDA_ERR(cudaMemcpy(dev_array, array, sizeof(int) * used_n, cudaMemcpyHostToDevice));

int main() {
    int n = 1000, m = 1000;
    Matrix* mtr = hilbert(n, m);
    // auto p = QRDecompositionNaive(mtr);
    // Matrix* res = multiply(p.first, p.second);
    auto trip = SVDDecomposition(mtr, 15, 1e-6);
    // printf("HI");
    auto tmp = multiply(trip->first, trip->second);
    auto t = transpose(trip->third);
    auto res = multiply(tmp, t);
    printf("Diff: %f \n", diff(mtr, res));
    // printf("Size(%d %d), Diff: %f \n", n, m, diff(mtr, res));
    // Matrix* d_mtr;
    // double* d_mtr_arr;
    // int* d_dims_arr;
    // copyMatrixFromHostToDevice(mtr, &d_mtr, &d_mtr_arr, &d_dims_arr);
    // show(mtr, n, m);
    // show(res, n, m);
    delete mtr;

    // printf("%d %d %d\n", d_mtr, d_mtr_arr, d_dims_arr);
    // (cudaFree(d_mtr));
    // (cudaFree(d_mtr_arr));
    // (cudaFree(d_dims_arr));
    delete tmp;
    delete t;
    delete res;
    delete trip->first;
    delete trip->second;
    delete trip->third;
    delete trip;
    return 0;
}
