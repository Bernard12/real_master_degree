//
// Created by ivan on 11.01.2020.
//

#include "svd.h"


double getBiggestEugenValueOfSquareMatrix(Matrix &a) {
    auto a_shape = a.shape();
    if (a_shape.first != a_shape.second) {
        return 0;
    }
    Matrix u = randomMatrix(a_shape.first, 1);
    for (int i = 0; i < 10; i++) {
        Matrix w = multiply(a, u);
        u = vectorColNormalize(w);
    }
    Matrix ut = transpose(u);
    Matrix tmp = multiply(a, u);
    return multiply(ut, tmp)[0][0];
}

pair<Matrix, Matrix> QRDecompositionNaive(Matrix &a) {
    auto a_shape = a.shape();
    int n = a_shape.first;
    int m = a_shape.second;
    Matrix Q(n, m), R(m, m);
    for (int i = 0; i < m; i++) {
        Matrix ai = subMatrix(a, 0, n + 0, i + 0, i + 1);
        for (int k = 0; k < i; k++) {
            Matrix qk = subMatrix(Q, 0, n + 0, k + 0, k + 1);
            Matrix qkt = transpose(qk);
            double v = -1 * multiply(qkt, ai)[0][0];
            Matrix tmp = multiply(qk, v);
            ai = sum(ai, tmp);
            R[k][i] = -1 * v;
        }
        R[i][i] = vectorColLength(ai);
        Matrix nai = vectorColNormalize(ai);
        for (int k = 0; k < n; k++) {
            Q[k][i] = nai[k][0];
        }
        Matrix qk = subMatrix(Q, 0, n + 0, i + 0, i + 1);
    }
#ifdef DEBUG
    Q.show();
    R.show();
#endif
    return make_pair(Q, R);
}

Triple SVDDecomposition(Matrix &a, int rank, double eps) {
    auto a_shape = a.shape();
    int n = a_shape.first;
    int m = a_shape.second;
    Matrix u = randomMatrix(n, rank), sgm = randomMatrix(rank, rank), v = randomMatrix(m, rank);
    Matrix at = transpose(a);
    double err = 1e9;
    for (; err > eps;) {
        Matrix av = multiply(a, v);
        auto qr_av = QRDecompositionNaive(av);
        u = subMatrix(qr_av.first, 0, n, 0, rank);
        Matrix atu = multiply(at, u);
        auto qr_atu = QRDecompositionNaive(atu);
        v = subMatrix(qr_atu.first, 0, m, 0, rank);
        sgm = subMatrix(qr_atu.second, 0, rank, 0, rank);
        // find error e = || A*V - U*SGM||
//        av = multiply(a, v);
        Matrix usgm = multiply(u, sgm);
        double revert = -1;
        usgm = multiply(usgm, revert);
        Matrix diff = sum(av, usgm);
        err = matrixNorm(diff);
    }
    return Triple(u, sgm, v);
}
