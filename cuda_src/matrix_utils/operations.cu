//
// Created by ivan on 11.01.2020.
//


#include "operations.cuh"

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
void multiply(Matrix *a, double b) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    int n = a->n;
    int m = a->m;
    for (int ij = startIdx; ij < n * m; ij+= offset) {
        a->matrix[ij] *= b;
    }
}

__global__
void equals(Matrix *a, Matrix *b, double eps, int* res) {
    const int n = a->n, m = a->m;
    int temp = 1;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->matrix[ij];
        double bij = b->matrix[ij];
        temp = temp & (abs(aij - bij) <= eps);
    }
    atomicAnd(res, temp);
}

/*
double diff(Matrix *a, Matrix *b) {
    Matrix* bla = multiply(b, -1);
    Matrix* res = sum(a, bla);
    auto shape = res->shape();
    double diff = 0;
    const int n = a->shape().first, m = a->shape().second;
    for (int ij = 0; ij < n * m; ij++) {
        double rij = res->get(ij / m, ij % m);
        diff = max(abs(diff), rij);
    }
    return diff;
}

Matrix* randomMatrix(int n, int m) {
    auto* res = new Matrix(n, m);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    for (int ij = 0; ij < n * m; ij++) {
        res->set(ij / m, ij % m, distribution(generator));
    }
    return res;
}

double vectorColLength(Matrix *a) {
    auto a_shape = a->shape();
    if (a_shape.second != 1) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < a_shape.first; i++) {
        double ai0 = a->get(i, 0);
        sum += ai0 * ai0;
    }
    return sqrt(sum);
}

double matrixNorm(Matrix *a) {
    auto a_shape = a->shape();
    const int n = a_shape.first, m = a_shape.second;
    double sum = 0;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        sum += aij * aij;
    }
    return sqrt(sum);
}

Matrix* vectorColNormalize(Matrix *a) {
    auto a_shape = a->shape();
    if (a_shape.second != 1) {
        return a;
    }
    double sum = 1 / vectorColLength(a);
    return multiply(a, sum);
}

Matrix* transpose(Matrix *a) {
    auto a_shape = a->shape();
    const int n = a_shape.first, m = a_shape.second;
    auto* res = new Matrix(a_shape.second, a_shape.first);
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        res->set(ij % m, ij / m, aij);
    }
    return res;
}

Matrix* subMatrix(Matrix *a, int rowStart, int rowEnd, int colStart, int colEnd) {
    int n = rowEnd - rowStart;
    int m = colEnd - colStart;
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("Bad shapes in submatrix");
    }
    auto* res = new Matrix(rowEnd - rowStart, colEnd - colStart);
    // TODO:
    // Add CUDA parallel option of loop
    for (int ij = 0; ij < n * m; ij++) {
        int i = ij / m;
        int j = ij % m;
        double aij = a->get(rowStart + i, colStart + j);
        res->set(i, j, aij);
    }
    return res;
}

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