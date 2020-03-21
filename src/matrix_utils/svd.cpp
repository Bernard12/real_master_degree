//
// Created by ivan on 11.01.2020.
//

#include "svd.h"


double getBiggestEugenValueOfSquareMatrix(Matrix *a) {
    auto a_shape = a->shape();
    if (a_shape.first != a_shape.second) {
        return 0;
    }
    auto u = randomMatrix(a_shape.first, 1);
    for (int i = 0; i < 10; i++) {
        auto w = multiply(a, u);
        u = vectorColNormalize(w);
    }
    auto ut = transpose(u);
    auto tmp = multiply(a, u);
    return multiply(ut, tmp)->get(0, 0);
}

pair<Matrix *, Matrix *> QRDecompositionNaive(Matrix *a) {
    auto a_shape = a->shape();
    int n = a_shape.first;
    int m = a_shape.second;
    auto Q = new Matrix(n, m), R = new Matrix(m, m);
    for (int i = 0; i < m; i++) {
        auto ai = subMatrix(a, 0, n + 0, i + 0, i + 1);
        for (int k = 0; k < i; k++) {
            auto qk = subMatrix(Q, 0, n + 0, k + 0, k + 1);
            auto qkt = transpose(qk);
            Matrix *tempMultiply = multiply(qkt, ai);
            double v = -1 * tempMultiply->get(0, 0);
            auto tmp = multiply(qk, v);
            Matrix *temp_ai = sum(ai, tmp);
            delete ai;
            ai = temp_ai;
            R->set(k, i, -1 * v);
            delete qk;
            delete qkt;
            delete tmp;
            delete tempMultiply;
        }
        R->set(i, i, vectorColLength(ai));
        auto nai = vectorColNormalize(ai);
        for (int k = 0; k < n; k++) {
            double nk0 = nai->get(k, 0);
            Q->set(k, i, isnan(nk0) ? 0 : nk0);
        }
//        Matrix* qk = subMatrix(Q, 0, n + 0, i + 0, i + 1);
        delete ai;
        delete nai;
    }
#ifdef DEBUG
    Q.show();
    R.show();
#endif
    return make_pair(Q, R);
}

Triple *SVDDecomposition(Matrix *a) {
    // LAPACKE_dgesvd();
    // A is (n, m) matrix in row-major
    int rows = a->shape().first, cols = a->shape().second;

    char jobu = 'S';
    char jobvt = 'S';

    // ld<somthing> is col size for COL_MAJOR and row size for ROW_MAJOR
    int lda = rows;

    int rank = min(rows, cols);

    auto *s = new double[rank];

    auto *u = new double[rows * rank];
    int ldu = rows;

    auto *vt = new double[rank * cols];
    int ldvt = rank;

    double superb[rank - 1];

    LAPACKE_dgesvd(
            LAPACK_COL_MAJOR,
            jobu, jobvt,
            rows, cols,
            a->matrix, lda,
            s,
            u, ldu,
            vt, ldvt,
            superb
    );

    Matrix *U = new Matrix(rows, rank);
    delete[] U->matrix;
    U->matrix = u;

    Matrix *VT = new Matrix(rank, cols);
    delete[] VT->matrix;
    VT->matrix = vt;

    Matrix *S = new Matrix(rank, rank);
    for (int i = 0; i < rank; i++) {
        S->set(i, i, s[i]);
    }
    delete[] s;

    return new Triple(U, S, VT);
}

Triple *SVDDecompositionNaive(Matrix *a, int rank, double eps) {
    auto a_shape = a->shape();
    int n = a_shape.first;
    int m = a_shape.second;
    auto u = randomMatrix(n, rank), sgm = randomMatrix(rank, rank), v = randomMatrix(m, rank);
    auto at = transpose(a);
    double err = 1e9;
    for (; err > eps;) {
        auto av = multiply(a, v);
        auto qr_av = QRDecompositionNaive(av);

        Matrix *u_tmp = subMatrix(qr_av.first, 0, n, 0, rank);
        delete u;
        u = u_tmp;

        auto atu = multiply(at, u);
        auto qr_atu = QRDecompositionNaive(atu);

        Matrix *v_tmp = subMatrix(qr_atu.first, 0, m, 0, rank);
        delete v;
        v = v_tmp;

        Matrix *sgm_tmp = subMatrix(qr_atu.second, 0, rank, 0, rank);
        delete sgm;
        sgm = sgm_tmp;

        // find error e = || A*V - U*SGM||
//        av = multiply(a, v);
        auto usgm = multiply(u, sgm);
        double revert = -1;
        Matrix *usgmt = multiply(usgm, revert);
        auto diff = sum(av, usgmt);
        err = matrixNorm(diff);

        delete av;
        delete qr_av.first;
        delete qr_av.second;
        delete atu;
        delete qr_atu.first;
        delete qr_atu.second;
        delete usgm;
        delete usgmt;
        delete diff;
    }
    delete at;
    return new Triple(u, sgm, v);
}

vector<Matrix *> TTDecomposition(Matrix *a, double eps) {
    vector<Matrix *> res;

    double norm = frobeniousMatrixNorm(a);
    double threshold = norm * eps / sqrt(a->shape_length - 1);

    int n_left = a->real_shape[0];
    int n_right = 1;
    for (int i = 1; i < a->shape_length; i++) {
        n_right *= a->real_shape[i];
    }

    vector<int> shape = {n_left, n_right};
    Matrix *M = a->copy();
    M->reshape(shape);

    // U S VT
    auto svd_m_full = SVDDecomposition(M);
    auto svd_m = trunkSVDResultsForTT(svd_m_full, eps);
    res.push_back(svd_m->first);
    int r = svd_m->second->real_shape[0];
    M = multiply(svd_m->second, svd_m->third);

    for (int i = 1; i < a->shape_length - 1; i++) {
        n_left = a->real_shape[i];
        n_right = n_right / a->real_shape[i];

        vector<int> next_shape = {r * n_left, n_right};
        M->reshape(next_shape);

        auto svd_m_next_full = SVDDecomposition(M);
        auto svd_m_next = trunkSVDResultsForTT(svd_m_next_full, eps);
        int r_cur = svd_m_next->second->real_shape[0];

        Matrix *GK = svd_m_next->first;
        vector<int> gk_shape = {r, n_left, r_cur};
        GK->reshape(gk_shape);

        res.push_back(GK);
        r = r_cur;

        M = multiply(svd_m_next->second, svd_m_next->third);
    }

    res.push_back(M);
    return res;
}

double getValueFromTrain(vector<Matrix *> m, vector<int> indexes) {
    double res = 0;
    Matrix *first = subMatrix(m[0], indexes[0], indexes[0] + 1, 0, m[0]->m());
    for (int i = 1; i < m.size() - 1; i++) {
        Matrix *cur = m[i]->get2DshapeFrom3d(indexes[i]);
        first = multiply(first, cur);
    }
    Matrix *last = subMatrix(m[m.size() - 1], 0, m[m.size() - 1]->n(), indexes[indexes.size() - 1],
                             indexes[indexes.size() - 1] + 1);
    first = multiply(first, last);
    return first->matrix[0];
}

// U S VT
Triple* trunkSVDResultsForTT(Triple* svd, double eps) {
    int rank = svd->second->n();
    double sum = 0;
    for (int i = rank - 1; i >= 1; i--) {
        double val = svd->second->get(i, i);
        if (sum + val > eps) {
            break;
        }
        sum += val;
        rank--;
    }
    Matrix* U = subMatrix(svd->first, 0, svd->first->n(), 0, rank);
    Matrix* S = subMatrix(svd->second, 0, rank, 0, rank);
    Matrix* VT = subMatrix(svd->third, 0, rank, 0, svd->third->m());
    return new Triple(U, S, VT);
}
