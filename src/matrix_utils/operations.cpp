//
// Created by ivan on 11.01.2020.
//


#include "operations.h"

Matrix sum(Matrix &a, Matrix &b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape != b_shape) {
        throw std::invalid_argument("Bad shapes");
    }
    Matrix res(a_shape.first, a_shape.second);
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < a_shape.second; j++) {
            res[i][j] = a[i][j] + b[i][j];
        }
    }
    return res;
}

Matrix multiply(Matrix &a, Matrix &b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.second != b_shape.first) {
        throw std::invalid_argument("Bad shapes");
    }
    Matrix res(a_shape.first, b_shape.second);
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < b_shape.second; j++) {
            for (int k = 0; k < a_shape.second; k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

Matrix multiply(Matrix &a, double &b) {
    auto a_shape = a.shape();
    Matrix res(a_shape.first, a_shape.second);
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < a_shape.second; j++) {
            res[i][j] = a[i][j] * b;
        }
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
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < a_shape.second; j++) {
            res = res && (abs(a[i][j] - b[i][j]) <= eps);
        }
    }
    return res;
}

Matrix randomMatrix(int n, int m) {
    Matrix res(n, m);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res[i][j] = distribution(generator);
        }
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

double matrixNorm(Matrix& a) {
    auto a_shape = a.shape();
    double sum = 0;
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < a_shape.second; j++) {
            sum += a[i][j] * a[i][j];
        }
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
    Matrix res(a_shape.second, a_shape.first);
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < a_shape.second; j++) {
            res[j][i] = a[i][j];
        }
    }
    return res;
}

Matrix subMatrix(Matrix &a, int rowStart, int rowEnd, int colStart, int colEnd) {
    int n = rowEnd - rowStart;
    int m = colEnd - colStart;
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("Bad shapes");
    }
    Matrix res(rowEnd - rowStart, colEnd - colStart);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            res[i][j] = a[rowStart + i][colStart + j];
        }
    }
    return res;
}

Matrix hilbert(int n, int m) {
    Matrix res(n, m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            res[i][j] = 1. / (i + j + 2);
        }
    }
    return res;
}
