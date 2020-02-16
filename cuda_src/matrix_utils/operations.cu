//
// Created by ivan on 11.01.2020.
//


#include "operations.cuh"

__host__ __device__
Matrix* sum(Matrix *a, Matrix *b) {
    const int n = a->n;
    const int m = a->m;
    const int nm = n * m;

    // int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // int offset = gridDim.x * blockDim.x;
    Matrix* res = new Matrix(n, m);

    for (int ij = 0; ij < nm; ij++) {
        double aij = a->matrix[ij];
        double bij = b->matrix[ij];
        res->matrix[ij] = aij + bij;
    }

    return res;
}

__host__ __device__
void show(Matrix* mtr, int n, int m) {
    // int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // int offset = gridDim.x * blockDim.x;
    printf("-----MATRIX(%d, %d)-------\n", n, m);
    // printf("StartIdx(%d), offset(%d)", startIdx, offset);
    for (int ij = 0; ij < n * m; ij++) {
        int i = ij / m, j = ij % m;

        if (j == 0) {
            printf("\n");
        }

        printf("%f ", mtr->get(i, j));
    }
    printf("\n-----------\n");
}

__host__ __device__
Matrix* multiply(Matrix *a, Matrix *b) {
    Matrix* res = new Matrix(a->n, b->m);
    // TODO
    // Added better implementation
    for (int i = 0; i < a->n; i++) {
        for (int j = 0; j < b->m; j++) {
            for (int k = 0; k < a->m; k++) {
                double aik = a->get(i, k);
                double bkj = b->get(k, j);
                double prev_resij = res->get(i, j);
                res->set(i, j, prev_resij + aik * bkj);
            }
        }
    }
    return res;
}

__host__ __device__
Matrix* multiply(Matrix *a, double b) {
    const int n = a->n, m = a->m;
    Matrix* res = new Matrix(n, m);

    for (int ij = 0; ij < n * m; ij++) {
        int i = ij / m;
        int j = ij % m;
        double aij = a->get(i, j);
        res->set(i, j, aij * b);
    }

    return res;
}

__global__
void multiply(Matrix* a, Matrix* b, Matrix* c) {
    int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;

    int n = a->n;
    int m = a->m;
    int k = b->m;


    for (int i = startIdx; i < n * k; i += offset) {
        double sum = 0;
        int row = i / k;
        int col = i % k;
        for(int q = 0; q < m; q++) {
            // sum += a->matrix[row * m + q] * b->matrix[q * m + col];
            sum += a->get(row, q) * b->get(q, col);
        }
        // c->matrix[row * k + col] = sum;
        c->set(row, col, sum);
    }
}

__host__ __device__
bool equals(Matrix *a, Matrix *b, double eps) {
    bool res = true;
    const int n = a->n, m = a->m;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m,ij % m);
        double bij = b->get(ij / m,ij % m);
        res = res && (abs(aij - bij) <= eps);
    }
    return res;
}

__host__ __device__
double diff(Matrix *a, Matrix *b) {
    Matrix* bla = multiply(b, -1);
    Matrix* res = sum(a, bla);
    double diff = 0;
    const int n = a->n, m = a->m;
    for (int ij = 0; ij < n * m; ij++) {
        double rij = res->get(ij / m, ij % m);
        diff = max(abs(diff), rij);
    }
    return diff;
}

__host__
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

__host__ __device__
double vectorColLength(Matrix *a) {
    double sum = 0;
    for (int i = 0; i < a->n; i++) {
        double ai0 = a->get(i, 0);
        sum += ai0 * ai0;
    }
    return sqrt(sum);
}

__host__ __device__
double matrixNorm(Matrix *a) {
    const int n = a->n, m = a->m;
    double sum = 0;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        sum += aij * aij;
    }
    return sqrt(sum);
}

__host__ __device__
Matrix* vectorColNormalize(Matrix *a) {
    double sum = 1 / vectorColLength(a);
    return multiply(a, sum);
}

__host__ __device__
Matrix* transpose(Matrix *a) {
    const int n = a->n, m = a->m;
    auto* res = new Matrix(m, n);
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        res->set(ij % m, ij / m, aij);
    }
    return res;
}

__host__ __device__
Matrix* subMatrix(Matrix *a, int rowStart, int rowEnd, int colStart, int colEnd) {
    int n = rowEnd - rowStart;
    int m = colEnd - colStart;
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

__host__ __device__
Matrix* hilbert(int n, int m) {
    auto* res = new Matrix(n, m);
    int i, j;
    // TODO:
    // Add CUDA parallel option of loop
    for (int ij = 0; ij < n * m; ij++) {
        i = ij / m;
        j = ij % m;
        res->set(ij / m, ij % m, 1. / (i + j + 2));
    }
    return res;
}