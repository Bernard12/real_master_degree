//
// Created by ivan on 11.01.2020.
//


#include "operations.h"
#include <omp.h>

// TODO: make research on cache misses!
Matrix sum(Matrix &a, Matrix &b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape != b_shape) {
        throw std::invalid_argument("Bad shapes in sum");
    }
    Matrix res(a_shape.first, a_shape.second);
//    TODO: add performance check -> single loop ~6% slower than nested
//    for (int i = 0; i < a_shape.first; i++) {
//        for (int j = 0; j < a_shape.second; j++) {
//            res[i][j] = a[i][j] + b[i][j];
//        }
//    }

    const int n = a_shape.first;
    const int m = a_shape.second;
    const int nm = n * m;
    int i, j;
    for (int ij = 0; ij < nm; ij++) {
        i = ij / m;
        j = ij % m;
        res[i][j] = a[i][j] + b[i][j];
    }
    return res;
}

Matrix multiply(Matrix &a, Matrix &b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.second != b_shape.first) {
        throw std::invalid_argument("Bad shapes in multiply");
    }
    Matrix res(a_shape.first, b_shape.second);
    // TODO
    // Added better implementation
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < b_shape.second; j++) {
            for (int k = 0; k < a_shape.second; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

Matrix multiply(Matrix &a, double b) {
    auto a_shape = a.shape();
    Matrix res(a_shape.first, a_shape.second);

    const int n = a_shape.first, m = a_shape.second;
    for (int ij = 0; ij < n * m; ij++) {
        res[ij / m][ij % m] = a[ij / m][ij % m] * b;
    }
    return res;
}

bool equals(Matrix &a, Matrix &b, double eps) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape != b_shape) {
        return false;
    }
    bool res = true;
    const int n = a_shape.first, m = a_shape.second;
    for (int ij = 0; ij < n * m; ij++) {
        res = res && (abs(a[ij / m][ij % m] - b[ij / m][ij % m]) <= eps);
    }
    return res;
}

double diff(Matrix &a, Matrix &b) {
    Matrix bla = multiply(b, -1);
    Matrix res = sum(a, bla);
    auto shape = res.shape();
    double diff = 0;
    const int n = a.shape().first, m = a.shape().second;
    for (int ij = 0; ij < n * m; ij++) {
        diff = max(abs(diff), res[ij / m][ij % m]);
    }
    return diff;
}

Matrix randomMatrix(int n, int m) {
    Matrix res(n, m);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    for (int ij = 0; ij < n * m; ij++) {
        res[ij / m][ij % m] = distribution(generator);
    }
    return res;
}

double vectorColLength(Matrix &a) {
    auto a_shape = a.shape();
    if (a_shape.second != 1) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < a_shape.first; i++) {
        sum += a[i][0] * a[i][0];
    }
    return sqrt(sum);
}

double matrixNorm(Matrix &a) {
    auto a_shape = a.shape();
    const int n = a_shape.first, m = a_shape.second;
    double sum = 0;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a[ij / m][ij % m];
        sum += aij * aij;
    }
    return sqrt(sum);
}

Matrix vectorColNormalize(Matrix &a) {
    auto a_shape = a.shape();
    if (a_shape.second != 1) {
        return a;
    }
    double sum = 1 / vectorColLength(a);
    return multiply(a, sum);
}

Matrix transpose(Matrix &a) {
    auto a_shape = a.shape();
    const int n = a_shape.first, m = a_shape.second;
    Matrix res(a_shape.second, a_shape.first);
    for (int ij = 0; ij < n * m; ij++) {
        res[ij % m][ij / m] = a[ij / m][ij % m];
    }
    return res;
}

Matrix subMatrix(Matrix &a, int rowStart, int rowEnd, int colStart, int colEnd) {
    int n = rowEnd - rowStart;
    int m = colEnd - colStart;
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("Bad shapes in submatrix");
    }
    Matrix res(rowEnd - rowStart, colEnd - colStart);
    // TODO:
    // Add CUDA parallel option of loop
    for (int ij = 0; ij < n * m; ij++) {
        res[ij / m][ij % m] = a[rowStart + ij / m][colStart + ij % m];
    }
    return res;
}

Matrix hilbert(int n, int m) {
    Matrix res(n, m);
    int i, j;
    // TODO:
    // Add CUDA parallel option of loop
    for (int ij = 0; ij < n * m; ij++) {
        i = ij / m;
        j = ij % m;
        res[ij / m][ij % m] = 1. / (i + j + 2);
    }
    return res;
}
