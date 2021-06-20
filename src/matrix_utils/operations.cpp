//
// Created by ivan on 11.01.2020.
//


#include "operations.h"
#include <omp.h>

Matrix *sum(Matrix *a, Matrix *b) {
    auto a_shape = a->shape();
    auto b_shape = b->shape();
    if (a_shape != b_shape) {
        throw std::invalid_argument("Bad shapes in sum");
    }
    auto res = new Matrix(a_shape.first, a_shape.second);

    const int n = a_shape.first;
    const int m = a_shape.second;
    const int nm = n * m;
    int i, j;
    for (int ij = 0; ij < nm; ij++) {
        i = ij / m;
        j = ij % m;
        double aij = a->get(i, j);
        double bij = b->get(i, j);
        res->set(i, j, aij + bij);
    }
    return res;
}

Matrix *multiply(Matrix *a, Matrix *b) {
    auto a_shape = a->shape();
    auto b_shape = b->shape();
    if (a_shape.second != b_shape.first) {
        throw std::invalid_argument("Bad shapes in multiply");
    }
    auto *res = new Matrix(a_shape.first, b_shape.second);
    // TODO
    // Added better implementation
    for (int i = 0; i < a_shape.first; i++) {
        for (int j = 0; j < b_shape.second; j++) {
            for (int k = 0; k < a_shape.second; k++) {
                double aik = a->get(i, k);
                double bkj = b->get(k, j);
                double prev_resij = res->get(i, j);
                res->set(i, j, prev_resij + aik * bkj);
            }
        }
    }
    return res;
}

Matrix *multiply(Matrix *a, double b) {
    auto a_shape = a->shape();
    auto *res = new Matrix(a_shape.first, a_shape.second);

    const int n = a_shape.first, m = a_shape.second;
    for (int ij = 0; ij < n * m; ij++) {
        int i = ij / m;
        int j = ij % m;
        double aij = a->get(i, j);
        res->set(i, j, aij * b);
    }
    return res;
}

bool equals(Matrix *a, Matrix *b, double eps) {
    auto a_shape = a->shape();
    auto b_shape = b->shape();
    if (a_shape != b_shape) {
        return false;
    }
    bool res = true;
    const int n = a_shape.first, m = a_shape.second;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        double bij = b->get(ij / m, ij % m);
        res = res && !isnan(aij) && !isnan(bij) && (abs(aij - bij) <= eps);
    }
    return res;
}

double diff(Matrix *a, Matrix *b) {
    Matrix *bla = multiply(b, -1);
    Matrix *res = sum(a, bla);
    auto shape = res->shape();
    double diff = 0;
    const int n = a->shape().first, m = a->shape().second;
    for (int ij = 0; ij < n * m; ij++) {
        double rij = res->get(ij / m, ij % m);
        diff = max(abs(diff), rij);
    }
    return diff;
}

Matrix *randomMatrix(int n, int m) {
    auto *res = new Matrix(n, m);
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

double frobeniousMatrixNorm(Matrix *a) {
    double sum = 0;

    for (int i = 0; i < a->total_element_count; i++) {
        double val = a->matrix[i];
        sum += val * val;
    }

    return sqrt(sum);
}

Matrix *vectorColNormalize(Matrix *a) {
    auto a_shape = a->shape();
    if (a_shape.second != 1) {
        return a;
    }
    double sum = 1 / vectorColLength(a);
    return multiply(a, sum);
}

Matrix *transpose(Matrix *a) {
    auto a_shape = a->shape();
    const int n = a_shape.first, m = a_shape.second;
    auto *res = new Matrix(a_shape.second, a_shape.first);
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        res->set(ij % m, ij / m, aij);
    }
    return res;
}

Matrix *subMatrix(Matrix *a, int rowStart, int rowEnd, int colStart, int colEnd) {
    int n = rowEnd - rowStart;
    int m = colEnd - colStart;
    if (n <= 0 || m <= 0) {
        throw std::invalid_argument("Bad shapes in submatrix");
    }
    auto *res = new Matrix(rowEnd - rowStart, colEnd - colStart);
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

Matrix *hilbert(int n, int m) {
    auto *res = new Matrix(n, m);
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

Matrix *hilbert(int n, int m, int k) {
    auto *res = new Matrix(n, m * k);
    vector<int> bla = {n, m, k};
    res->reshape(bla);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int q = 0; q < k; q++) {
                int index = i + n * j + n * m * q;
                double val = 1. / (i + j + q + 1);
                res->matrix[index] = val;
            }
        }
    }
    return res;
}

Matrix *hilbert(int i1, int i2, int i3, int i4, int i5) {
    auto *res = new Matrix(i1, i2 * i3 * i4 * i5);
    vector<int> bla = {i1, i2, i3, i4, i5};
    res->reshape(bla);
    for (int i = 0; i < i1; i++) {
        for (int j = 0; j < i2; j++) {
            for (int k = 0; k < i3; k++) {
                for (int l = 0; l < i4; l++) {
                    for (int w = 0; w < i5; w++) {
                        int index =
                                i +
                                i1 * j +
                                i1 * i2 * k +
                                i1 * i2 * i3 * l +
                                i1 * i2 * i3 * i4 * w;
                        double val = 1. / (i + j + k + l + w + 1);
                        res->matrix[index] = val;
                    }
                }
            }
        }
    }
    return res;
}

Matrix *hilbert(int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
    auto *res = new Matrix(i1, i2 * i3 * i4 * i5 * i6 * i7);
    vector<int> bla = {i1, i2, i3, i4, i5, i6, i7};
    res->reshape(bla);
    for (int i = 0; i < i1; i++) {
        for (int j = 0; j < i2; j++) {
            for (int k = 0; k < i3; k++) {
                for (int l = 0; l < i4; l++) {
                    for (int w = 0; w < i5; w++) {
                        for (int e = 0; e < i7; e++) {
                            for (int r = 0; r < i7; r++) {
                                int index =
                                        i +
                                        i1 * j +
                                        i1 * i2 * k +
                                        i1 * i2 * i3 * l +
                                        i1 * i2 * i3 * i4 * w +
                                        i1 * i2 * i3 * i4 * i5 * e +
                                        i1 * i2 * i3 * i4 * i5 * i6 * r;

                                double val = 1. / (i + j + k + l + w + e + r + 1);
                                res->matrix[index] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}

/**
 * @param r - [0,1] split
 * @param d - dimensions
 * @return
 */
Matrix *sinCube(int r, double step) {
    int d = 7;

    vector<int> shape;
    int total = 1;
    for (int i = 0; i < d; i++) {
        shape.push_back(r);
        total *= r;
    }
    total /= r;


    auto *res = new Matrix(r, total);
    res->reshape(shape);

    for (int i1 = 0; i1 < r; i1++) {
        for (int i2 = 0; i2 < r; i2++) {
            for (int i3 = 0; i3 < r; i3++) {
                for (int i4 = 0; i4 < r; i4++) {
                    for (int i5 = 0; i5 < r; i5++) {
                        for (int i6 = 0; i6 < r; i6++) {
                            for (int i7 = 0; i7 < r; i7++) {
                                double sum =
                                        i1 * step + i2 * step +
                                        i3 * step + i4 * step +
                                        i5 * step + i6 * step +
                                        i7 * step;
                                double sn = sin(sum);
                                int index =
                                        i1 +
                                        r * i2 +
                                        r * r * i3 +
                                        r * r * r * i4 +
                                        r * r * r * r * i5 +
                                        r * r * r * r * r * i6 +
                                        r * r * r * r * r * r * i7;
                                res->matrix[index] = sn;
                            }
                        }
                    }
                }
            }
        }
    }

    return res;
}

double convolution(vector<Matrix *> tt, vector<Matrix *> u) {
    auto v1 = multiply(transpose(u[0]), tt[0]);
    auto vn = multiply(tt.back(), u.back());

    vector<Matrix *> gks;

    for (int i = 1; i < tt.size() - 1; i++) {
        int r1 = tt[i]->real_shape[0];
        int nk = tt[i]->real_shape[1];
        int r2 = tt[i]->real_shape[2];

        auto *gk_cur = new Matrix(r1, r2);

        for (int i1 = 0; i1 < r1; i1++) {
            for (int i2 = 0; i2 < r2; i2++) {
                for (int q = 0; q < nk; q++) {
                    double val = gk_cur->get(i1, i2);

                    double sum = tt[i]->matrix[i1 + r1 * q + r1 * nk * i2] * u[i]->get(q, 0);

                    gk_cur->set(i1, i2, val + sum);
                }
            }
        }

        gks.push_back(gk_cur);
    }

    Matrix *v = v1;

    for (int i = 0; i < gks.size(); i++) {
        v = multiply(v, gks[i]);
    }

    v = multiply(v, vn);

    return v->get(0, 0);
}