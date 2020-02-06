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
    Matrix* m = randomMatrix(322, 228);
    auto qr = QRDecompositionNaive(m);
    Matrix* mt = multiply(qr.first, qr.second);
    double eps = precisions["strong"];
    ASSERT_TRUE(equals(m, mt, eps));
}
