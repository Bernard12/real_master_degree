//
// Created by ivan on 26.01.2020.
//

#include "gtest/gtest.h"
#include "../../src/matrix/Matrix.h"
#include "../../src/matrix_utils/operations.h"
#include "../../src/matrix_utils/svd.h"
#include <map>

using namespace std;

static map<string, double> precisions = {
        {"weak",   1e-3},
        {"medium", 1e-6},
        {"strong", 1e-9},
};

TEST(qr_naive, test_sing_matrix_1) {
    Matrix* m = new Matrix(2, 2);
    m->set(0, 0, 1);
    m->set(0, 1, 2);
    m->set(1, 0, 2);
    m->set(1, 1, 4);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["weak"];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double aij = mt->get(i, j);
            ASSERT_FALSE(isnan(aij));
        }
    }
    ASSERT_LE(diff(m, mt), eps);
}

TEST(qr_naive, test_sing_matrix_2) {
    int n = 3, mm = 3;
    Matrix* m = new Matrix(n, mm);
    m->set(0, 0, 1);
    m->set(0, 1, 2);
    m->set(0, 2, 3);
    m->set(1, 0, 2);
    m->set(1, 1, 4);
    m->set(1, 2, 7);
    m->set(2, 0, 3);
    m->set(2, 1, 6);
    m->set(2, 2, 9);
    auto qr = QRDecompositionNaive(m);
    qr.first->show();
    qr.second->show();
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["weak"];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < mm; j++) {
            double aij = mt->get(i, j);
            ASSERT_FALSE(isnan(aij));
        }
    }
    ASSERT_LE(diff(m, mt), eps);
}

TEST(qr_naive, test_small_matrix_weak) {
    Matrix* m = randomMatrix(2, 2);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["weak"];
    ASSERT_LE(diff(m, mt), eps);
}

TEST(qr_naive, test_small_matrix_medium) {
    Matrix* m = randomMatrix(2, 2);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["medium"];
    ASSERT_TRUE(equals(m, mt, eps));
}

TEST(qr_naive, test_small_matrix_strong) {
    Matrix* m = randomMatrix(2, 2);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["strong"];
    ASSERT_TRUE(equals(m, mt, eps));
}

TEST(qr_naive, test_medium_matrix_weak) {
    Matrix* m = randomMatrix(25, 14);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["weak"];
    ASSERT_TRUE(equals(m, mt, eps));
}

TEST(qr_naive, test_medium_matrix_medium) {
    Matrix* m = randomMatrix(25, 14);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["medium"];
    ASSERT_TRUE(equals(m, mt, eps));
}

TEST(qr_naive, test_medium_matrix_strong) {
    Matrix* m = randomMatrix(25, 14);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["strong"];
    ASSERT_TRUE(equals(m, mt, eps));
}


TEST(qr_naive, test_big_matrix_weak) {
    Matrix* m = randomMatrix(322, 228);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["weak"];
    ASSERT_TRUE(equals(m, mt, eps));
}

TEST(qr_naive, test_big_matrix_medium) {
    Matrix* m = randomMatrix(322, 228);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["medium"];
    ASSERT_TRUE(equals(m, mt, eps));
}

TEST(qr_naive, test_big_matrix_strong) {
    Matrix* m = randomMatrix(228, 322);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["strong"];
    ASSERT_TRUE(equals(m, mt, eps));
}
