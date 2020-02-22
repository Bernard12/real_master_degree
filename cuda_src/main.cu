#include "matrix/Matrix.cuh"
#include "matrix_utils/svd.cuh"
#include <stdio.h>

// CATCH_CUDA_ERR(cudaMalloc(&dev_array, sizeof(int) * used_n));
// CATCH_CUDA_ERR(cudaMemcpy(dev_array, array, sizeof(int) * used_n, cudaMemcpyHostToDevice));

int main() {
    int n = 10000, m = 10000;
    Matrix* mtr = hilbert(n, m);
    // auto p = QRDecompositionNaive(mtr);
    // Matrix* res = multiply(p.first, p.second);
    auto trip = SVDDecomposition(mtr, 10, 1e-6);
    auto tmp = multiply(trip->first, trip->second);
    auto t = transpose(trip->third);
    auto res = multiply(tmp, t);
    // printf("Diff: %f \n", diff(mtr, res));
    printf("Size(%d %d), Diff: %f \n", n, m, diff(mtr, res));
    // show(mtr, n, m);
    // show(res, n, m);
    delete mtr;
    delete tmp;
    delete t;
    delete res;
    delete trip->first;
    delete trip->second;
    delete trip->third;
    delete trip;
    return 0;
}
