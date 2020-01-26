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