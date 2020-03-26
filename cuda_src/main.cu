#include "matrix/Matrix.cuh"
#include "matrix_utils/svd.cuh"
#include <stdio.h>

// CATCH_CUDA_ERR(cudaMalloc(&dev_array, sizeof(int) * used_n));
// CATCH_CUDA_ERR(cudaMemcpy(dev_array, array, sizeof(int) * used_n, cudaMemcpyHostToDevice));

int main() {
    int n = 4, m = 16, k = 4;
    Matrix* mtr = hilbert(n,m,k);
    // Matrix* mtr = hilbert(m,n);
    // Matrix* mtr = hilbert(n,m);
    // int * shapes = new int[3];
    // shapes[0] = n;
    // shapes[1] = m;
    // shapes[2] = k;
    // Matrix* mtr = hilbert(n, m, k);
    // delete[] shapes;
    // auto p = QRDecompositionNaive(mtr);
    // Matrix* res = multiply(p.first, p.second);
    // auto trip = SVDDecomposition(mtr, 10, 1e-6);
    // auto svd = SVDDecompositionwCUB(mtr);
    // show(svd->first, 4, 4);
    // show(svd->second, 4, 4);
    // show(svd->third, 4, 16);
    // printf("HI");
    // auto tmp = multiply(svd->first, svd->second);
    // auto res = multiply(tmp, svd->third);

    // auto tmp = multiply(transpose(svd->third), svd->second);
    // auto res = multiply(tmp, transpose(svd->first));
    // printf("Diff: %f \n", diff(mtr, res));
    // show(res, 4, 16);
    // show(res, 16, 4);
    // printf("Size(%d %d), Diff: %f \n", n, m, diff(mtr, res));
    // Matrix* d_mtr;
    // double* d_mtr_arr;
    // int* d_dims_arr;
    // copyMatrixFromHostToDevice(mtr, &d_mtr, &d_mtr_arr, &d_dims_arr);
    // show(mtr, n, m);
    // int* newShapes = new int[2];
    // newShapes[0] = 8;
    // newShapes[1] = 2;
    // mtr->reshape(newShapes, 2);
    // show(mtr, 8, 2);
    printf("start tt decomposition %d\n", mtr->shape_length);
    auto tt = TTDecomposition(mtr, 1e-6);
    // show(tt[2], 4, 4);
    for(auto i : tt) {
        printf("%d ", i->shape_length);
    }
    printf("\n");
    vector<int> indexes = {0, 1, 0};
    double val = getValueFromTrain(tt, indexes);
    printf("Value: %f\n", val);

    // show(res, n, m);
    // show(mtr, n, m, k);
    // tensorTrain(mtr, 1e-6);
    // delete mtr;

    // printf("%d %d %d\n", d_mtr, d_mtr_arr, d_dims_arr);
    // (cudaFree(d_mtr));
    // (cudaFree(d_mtr_arr));
    // (cudaFree(d_dims_arr));
    // delete tmp;
    // delete t;
    // delete res;
    // delete trip->first;
    // delete trip->second;
    // delete trip->third;
    // delete trip;
    return 0;
}
