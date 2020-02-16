//
// Created by ivan on 11.01.2020.
//


#include "operations.cuh"

__global__
void get(Matrix* mtr, int index, double* res) {
    *res = mtr->matrix[index];
}

__global__
void set(Matrix* mtr, int index, double* res) {
    mtr->matrix[index] = *res;
}

__global__
void sum(Matrix *a, Matrix *b) {
    const int n = a->n;
    const int m = a->m;
    const int nm = n * m;

    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    for (int ij = startIdx; ij < nm; ij += offset) {
        double aij = a->matrix[ij];
        double bij = b->matrix[ij];
        a->matrix[ij] = aij + bij;
    }
}

__global__
void show(Matrix* mtr) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int n = mtr->n;
    int m = mtr->m;
    printf("-----MATRIX(%d, %d)-------\n", n, m);
    printf("StartIdx(%d), offset(%d)", startIdx, offset);
    for (int ij = startIdx; ij < n * m; ij+= offset) {
        if (ij % m == 0) {
            printf("\n");
        }
        printf("%f ", mtr->matrix[ij]);
    }
    printf("\n-----------\n");
}

__global__
void multiply(Matrix *a, Matrix *b, Matrix* res) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    int n = a->n;
    int m = a->m;
    int k = b->m;

    double sum = 0;

    for (int i = startIdx; i < n * k; i += offset) {
        int row = i / k;
        int col = i % k;
        for(int q = 0; q < m; q++) {
            sum += a->matrix[row * m + q] * b->matrix[q * m + col];
        }
        res->matrix[row * k + col] = sum;
    }
}

__global__
void multiply(Matrix *a, double* b) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int n = a->n;
    int m = a->m;
    for (int ij = startIdx; ij < n * m; ij+= offset) {
        a->matrix[ij] *= (*b);
    }
}

__global__
void equals(Matrix *a, Matrix *b, double eps, int* res) {
    const int n = a->n, m = a->m;
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int temp = 1;
    for (int ij = startIdx; ij < n * m; ij += offset) {
        double aij = a->matrix[ij];
        double bij = b->matrix[ij];
        temp = temp & (abs(aij - bij) <= eps);
    }
    atomicAnd(res, temp);
}

void diff(Matrix *a, Matrix *b, double* diff) {
    const int n = a->n, m = a->m;
    int max_diff = 0;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->matrix[ij];
        double bij = b->matrix[ij];
        max_diff = abs(aij - bij) > max_diff ? abs(aij - bij) : max_diff;
    }
    *diff = max_diff;
}

void randomMatrix(int n, int m, Matrix* res) {
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    for (int ij = 0; ij < n * m; ij++) {
        res->matrix[ij] = distribution(generator);
    }
}

void vectorColLength(Matrix *a, double* res) {
    double sum = 0;
    for (int i = 0; i < a->n; i++) {
        // double ai0 = a->get(i, 0);
        double ai0 = a->matrix[i * a->m];
        sum += ai0 * ai0;
    }
    *res = sqrt(sum);
}

void matrixNorm(Matrix *a, double* res) {
    const int n = a->n, m = a->m;
    double sum = 0;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->matrix[ij];
        sum += aij * aij;
    }
    *res = sqrt(sum);
}

void vectorColNormalize(Matrix *a) {
    double len = 0;
    // TODO: need sync to work
    vectorColLength(a, &len);
    double sum = 1.0 / len;
    multiply<<<16, 16>>>(a, &sum);
}

__host__ __device__
void transpose(Matrix* a, Matrix* res) {
    int n = a->n, m = a->m;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->matrix[ij];
        int i = ij / m;
        int j = ij % m;
        res->matrix[j * n + i] = aij;
    }
}

__host__ __device__
void subMatrix(Matrix* a, int rowStart, int rowEnd, int colStart, int colEnd, Matrix* res) {
    int ij = 0;
    for (int i = rowStart; i < rowEnd; i++) {
        for (int j = colStart; j < colEnd; j++) {
            res->matrix[ij++] = a->matrix[i * a->m + j];
        }
    }
}

/*

void hilbert(int n, int m, Matrix* res) {
    // auto* res = new Matrix(n, m);
    int i, j;
    // TODO:
    // Add CUDA parallel option of loop
    for (int ij = 0; ij < n * m; ij++) {
        i = ij / m;
        j = ij % m;
        res->set(ij / m, ij % m, 1. / (i + j + 2));
    }
}

*/