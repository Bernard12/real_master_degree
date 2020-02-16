#include "./svd.cuh"

pair<Matrix*, Matrix*> QRDecompositionNaive(Matrix *a) {
    int n = a->n;
    int m = a->m;
    auto Q = new Matrix(n, m), R = new Matrix(m, m);
    for (int i = 0; i < m; i++) {
        auto ai = subMatrix(a, 0, n + 0, i + 0, i + 1);
        for (int k = 0; k < i; k++) {
            auto qk = subMatrix(Q, 0, n + 0, k + 0, k + 1);
            auto qkt = transpose(qk);
            Matrix* tempMultiply = multiply(qkt, ai);
            double v = -1 * tempMultiply->get(0, 0);
            auto tmp = multiply(qk, v);
            Matrix* temp_ai = sum(ai, tmp);
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
            Q->set(k, i, nk0);
        }
        delete ai;
        delete nai;
    }
#ifdef DEBUG
    Q.show();
    R.show();
#endif
    return make_pair(Q, R);
}

Triple* SVDDecomposition(Matrix *a, int rank, double eps) {
    int n = a->n;
    int m = a->m;
    auto u = randomMatrix(n, rank), sgm = randomMatrix(rank, rank), v = randomMatrix(m, rank);
    auto at = transpose(a);
    double err = 1e9;
    for (; err > eps;) {
        auto av = multiply(a, v);
        auto qr_av = QRDecompositionNaive(av);

        Matrix* u_tmp = subMatrix(qr_av.first, 0, n, 0, rank);
        delete u;
        u = u_tmp;

        auto atu = multiply(at, u);
        auto qr_atu = QRDecompositionNaive(atu);

        Matrix* v_tmp = subMatrix(qr_atu.first, 0, m, 0, rank);
        delete v;
        v = v_tmp;

        Matrix* sgm_tmp = subMatrix(qr_atu.second, 0, rank, 0, rank);
        delete sgm;
        sgm = sgm_tmp;

        // find error e = || A*V - U*SGM||
//        av = multiply(a, v);
        auto usgm = multiply(u, sgm);
        double revert = -1;
        Matrix* usgmt = multiply(usgm, revert);
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