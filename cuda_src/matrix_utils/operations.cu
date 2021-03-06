//
// Created by ivan on 11.01.2020.
//


#include "operations.cuh"

__host__ __device__
Matrix* sum(Matrix *a, Matrix *b) {
    const int n = a->n();
    const int m = a->m();
    const int nm = n * m;

    // int startIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // int offset = gridDim.x * blockDim.x;
    Matrix* res = new Matrix(n, m);

    for (int ij = 0; ij < nm; ij++) {
        int i = ij / m, j = ij % m;
        double aij = a->get(i, j);
        double bij = b->get(i, j);
        res->set(i, j, aij + bij);
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

__host__
void show(Matrix* mtr, int n, int m, int k) {

    printf("-----TENSOR(%d, %d, %d)-------\n", n, m, k);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int q = 0; q < k; q++) {
                printf("(%d, %d, %d): %f\n", i, j, q, mtr->get(i, j, k));
            }
        }
    }

}

__host__ __device__
Matrix* multiply(Matrix *a, Matrix *b) {
    Matrix* res = new Matrix(a->n(), b->m());
    // TODO
    // Added better implementation
    for (int i = 0; i < a->n(); i++) {
        for (int j = 0; j < b->m(); j++) {
            for (int k = 0; k < a->m(); k++) {
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
    const int n = a->n(), m = a->m();
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

    int n = a->n();
    int m = a->m();
    int k = b->m();


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
    const int n = a->n(), m = a->m();
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m,ij % m);
        double bij = b->get(ij / m,ij % m);
        res = res && (abs(aij - bij) <= eps);
    }
    return res;
}

__host__ __device__
double diff(Matrix *a, Matrix *b) {
    Matrix* tmp = multiply(b, -1);
    Matrix* res = sum(a, tmp);
    double diff = 0;
    const int n = a->n(), m = a->m();
    for (int ij = 0; ij < n * m; ij++) {
        double rij = res->get(ij / m, ij % m);
        diff = max(abs(diff), rij);
    }
    delete tmp;
    delete res;
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
    for (int i = 0; i < a->n(); i++) {
        double ai0 = a->get(i, 0);
        sum += ai0 * ai0;
    }
    return sqrt(sum);
}

__host__ __device__
double matrixNorm(Matrix *a) {
    const int n = a->n(), m = a->m();
    double sum = 0;
    for (int ij = 0; ij < n * m; ij++) {
        double aij = a->get(ij / m, ij % m);
        sum += aij * aij;
    }
    return sqrt(sum);
}

__host__ __device__
double frobeniousNorm(Matrix* a) {
    int total = 1;
    for (int i = 0; i < a->shape_length; i++) {
        total *= a->real_shape[i];
    }
    double sum = 0;
    for (int i = 0; i < total; i++) {
        double value = a->matrix[i];
        sum += value * value;
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
    const int n = a->n(), m = a->m();
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
        res->set(ij / m, ij % m, 1. / (i + j + 1));
    }
    return res;
}

__host__
Matrix* hilbert(int n, int m, int k) {
    int * shapes = new int[3];
    shapes[0] = n;
    shapes[1] = m;
    shapes[2] = k;
    auto* res = new Matrix(n * m, k);
    res->reshape(shapes, 3);
    delete[] shapes;
    // TODO:
    // Add CUDA parallel option of loop
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            for (int q = 0; q < k; q++) {
                double val = 1./(i + j + q + 1);
                // printf("Set(%d, %d, %d) %f\n", i, j, q, val);
                int index = i + n * j + n * m * q;
                res->matrix[index] = val;
            }
        }
    }
    return res;
}

Matrix* hilbert(int i1, int i2, int i3, int i4, int i5) {
    int* shapes = new int[5]{i1, i2, i3, i4, i5};
    auto* res = new Matrix(i1, i2 * i3 * i4 * i5);
    res->reshape(shapes, 5);
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

Matrix* hilbert(int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
    int* shapes = new int[7]{i1, i2, i3, i4, i5, i6, i7};
    auto* res = new Matrix(i1, i2 * i3 * i4 * i5 * i6 * i7);
    res->reshape(shapes, 7);
    for (int i = 0; i < i1; i++) {
        for (int j = 0; j < i2; j++) {
            for (int k = 0; k < i3; k++) {
                for (int l = 0; l < i4; l++) {
                    for (int w = 0; w < i5; w++) {
                        for (int e = 0; e < i6; e++) {
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

    int* next_shape = new int[shape.size()];
    for ( int i = 0; i < shape.size(); i++) {
        next_shape[i] = shape[i];
    }

    res->reshape(next_shape, shape.size());

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
                                        i7 * step ;
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

double convolution(vector<Matrix*> tt, vector<Matrix *> u) {
    auto v1 = multiply(transpose(u[0]), tt[0]);
    auto vn = multiply(tt.back(), u.back());

    vector<Matrix*> gks;

    for (int i = 1; i < tt.size() - 1; i++) {
        int r1 = tt[i]->real_shape[0];
        int nk = tt[i]->real_shape[1];
        int r2 = tt[i]->real_shape[2];

        auto* gk_cur = new Matrix(r1, r2);

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

    Matrix* v = v1;

    for (int i = 0; i < gks.size(); i++) {
        v = multiply(v, gks[i]);
    }

    v = multiply(v, vn);

    return v->get(0, 0);
}