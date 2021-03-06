//
// Created by ivan on 30.01.2020.
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

TEST(svd_naive, test_small_matrix_easy) {
    Matrix* m = new Matrix(2, 2);
    m->set(0, 0, 3);
    m->set(0, 1, 0);
    m->set(1, 0, 4);
    m->set(1, 1, 5);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 2, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    Matrix* vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_small_matrix_weak) {
    Matrix* m = randomMatrix(2, 2);
    double eps = precisions["weak"];
    auto svd = SVDDecomposition(m, 2, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_small_matrix_medium) {
    Matrix* m = randomMatrix(2, 2);
    double eps = precisions["medium"];
    auto svd = SVDDecomposition(m, 2, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_small_matrix_strong) {
    Matrix* m = randomMatrix(2, 2);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 2, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_big_matrix_medium) {
    Matrix* m = randomMatrix(30, 30);
    double eps = precisions["medium"];
    auto svd = SVDDecomposition(m, 30, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_big_matrix_strong) {
    Matrix* m = randomMatrix(30, 30);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 30, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_small_hilbert_matrix_strong) {
    Matrix* m = hilbert(2, 2);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 2, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}


TEST(svd_naive, test_big_hilbert_matrix_strong) {
    Matrix* m = hilbert(15, 15);
    double eps = precisions["weak"];
    auto svd = SVDDecomposition(m, 10, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_extra_big_hilbert_matrix_strong) {
    Matrix* m = hilbert(400, 400);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 10, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_extra_long_hilbert_matrix_strong) {
    Matrix* m = hilbert(400, 4000);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 10, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}

TEST(svd_naive, test_extra_big_hilbert_matrix) {
    Matrix* m = hilbert(4000, 4000);
    double eps = precisions["strong"];
    auto svd = SVDDecomposition(m, 10, eps);
    Matrix* u_sgm = multiply(svd->first, svd->second);
    auto vt = transpose(svd->third);
    Matrix* res = multiply(u_sgm, vt);
    double d = diff(m, res);
    delete m;
    delete u_sgm;
    delete res;
    ASSERT_LE(d, 1e-3);
}
