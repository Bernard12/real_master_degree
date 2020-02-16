#include "./svd.cuh"

#define CCE(errValue)                                                   \
    do {                                                                \
        if (errValue != cudaSuccess) {                                  \
            fprintf(stderr ,"[CUDA-ERROR]-[%s(line:%d)]: %s\n", __FILE__, __LINE__, cudaGetErrorString(errValue)); \
            exit(0);                                                    \
        }                                                               \
    } while(0);

// CATCH_CUDA_ERR(cudaMalloc(&dev_array, sizeof(int) * used_n));
// CATCH_CUDA_ERR(cudaMemcpy(dev_array, array, sizeof(int) * used_n, cudaMemcpyHostToDevice));

void copyMatrixFromHostToDevice(Matrix* hostMatrix, Matrix** deviceMatrix) {
    const int n = hostMatrix->n, m = hostMatrix->m;
    Matrix* temp = new Matrix(n, m);

    const int matrix_size = sizeof(double) * n * m;
    CCE(cudaMalloc(&temp->matrix, matrix_size));
    CCE(cudaMemcpy(temp->matrix, hostMatrix->matrix, sizeof(Matrix) * 1, cudaMemcpyHostToDevice));

    CCE(cudaMalloc(deviceMatrix, sizeof(Matrix) * 1));
    CCE(cudaMemcpy(*deviceMatrix, temp, sizeof(Matrix) * 1, cudaMemcpyHostToDevice));

    temp->matrix = NULL;
    delete temp;
}

/*
 * Matrix* a - matrix on host device
 * @return - 2 matrix on device
 */
pair<Matrix*, Matrix*> QRDecompositionNaive(Matrix *a) {
    int n = a->n;
    int m = a->m;

    /* ATTENTION PLEASE */
    /* MATRIX HoST */
    Matrix* Q = new Matrix(n, m);
    Matrix* R = new Matrix(m, m);
    Matrix* ai = new Matrix(n, 1);
    Matrix* qk = new Matrix(n, 1);
    Matrix* qkt = new Matrix(1, n);
    Matrix* tempMultiply = new Matrix(1, 1);
    /* ATTENTION PLEASE */
    Matrix* a_dev;
    copyMatrixFromHostToDevice(a, &a_dev);
    Matrix* Q_dev;
    copyMatrixFromHostToDevice(Q, &Q_dev);
    Matrix* R_dev;
    copyMatrixFromHostToDevice(R, &R_dev);
    Matrix* ai_dev;
    copyMatrixFromHostToDevice(ai, &ai_dev);
    Matrix* qk_dev;
    copyMatrixFromHostToDevice(qk, &qk_dev);
    Matrix* qkt_dev;
    copyMatrixFromHostToDevice(qkt, &qkt_dev);
    Matrix* tempMultiply_dev;
    copyMatrixFromHostToDevice(tempMultiply, &tempMultiply_dev);
    /* ATTENTION PLEASE */
    for (int i = 0; i < m; i++) {
        subMatrix(a_dev, 0, n + 0, i + 0, i + 1, ai_dev);
        for (int k = 0; k < i; k++) {
            subMatrix(Q_dev, 0, n + 0, k + 0, k + 1, qk_dev);
            transpose(qk_dev, qkt_dev);
            multiply<<<1, 1>>>(qkt_dev, ai_dev, tempMultiply_dev);
            CCE(cudaGetLastError());
            double* v;
            CCE(cudaMalloc(&v, sizeof(double)));
            get<<<1, 1>>>(tempMultiply_dev, 0, v);
            CCE(cudaGetLastError());
            // tempmultiply -> get double
            multiply<<<1, 1>>>(qk, v);
            CCE(cudaGetLastError());
            sum<<<1, 1>>>(ai, qk);
            CCE(cudaGetLastError());
            // Attention should use some kind of kernel
            // update<<<1, 1>>>(Matrix*, int index, double value)
            // R->matrix[k * m + i] = -1 * v;
            set<<<1, 1>>>(R, k * m + i, v);
            CCE(cudaGetLastError());
            // R->set(k, i, -1 * v);
        }
        vectorColLength(ai, &(R->matrix[i * m + i]));
        // todo: fix it
        vectorColNormalize(ai);
        for (int k = 0; k < n; k++) {
            double nk0 = ai->matrix[k];
            Q->matrix[k * m + i] = nk0;
        }
//        Matrix* qk = subMatrix(Q, 0, n + 0, i + 0, i + 1);
    }
#ifdef DEBUG
    Q.show();
    R.show();
#endif
    return make_pair(Q, R);
}

/*

Triple* SVDDecomposition(Matrix *a, int rank, double eps) {
    auto a_shape = a->shape();
    int n = a_shape.first;
    int m = a_shape.second;
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
*/
