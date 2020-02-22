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

void copyMatrixFromHostToDevice(Matrix* hostMatrix, Matrix** deviceMatrix, double** deviceMatrixArray) {
    const int n = hostMatrix->n, m = hostMatrix->m;
    Matrix* temp = new Matrix(n, m);

    const int matrix_size = sizeof(double) * n * m;
    CCE(cudaMalloc(&temp->matrix, matrix_size));
    CCE(cudaMemcpy(temp->matrix, hostMatrix->matrix, matrix_size, cudaMemcpyHostToDevice));

    CCE(cudaMalloc(deviceMatrix, sizeof(Matrix) * 1));
    CCE(cudaMemcpy(*deviceMatrix, temp, sizeof(Matrix) * 1, cudaMemcpyHostToDevice));

    *deviceMatrixArray = temp->matrix;
    temp->matrix = NULL;
    delete temp;
}

void copyMatrixFromDeviceToHost(double* deviceMatrixArray, Matrix** hostMatrix, int n, int m) {
    *hostMatrix = new Matrix(n, m);
    CCE(
        cudaMemcpy(
            (*hostMatrix)->matrix,
            deviceMatrixArray,
            sizeof(double) * n * m,
            cudaMemcpyDeviceToHost
        )
    );
}

/*
    QRDecompostion is part of SVD and should be called from host
    Matrix* a - pointer to matrix on host
    @return pair of Q and R matrix on host
*/
pair<Matrix*, Matrix*> QRDecompositionNaive(Matrix *a) {
    int n = a->n;
    int m = a->m;
    Matrix* Q = new Matrix(n, m);
    Matrix* R = new Matrix(m, m);
    for (int i = 0; i < m; i++) {
        auto ai = subMatrix(a, 0, n + 0, i + 0, i + 1);
        for (int k = 0; k < i; k++) {
            Matrix* qk = subMatrix(Q, 0, n + 0, k + 0, k + 1);
            Matrix* qkt = transpose(qk);
            Matrix* tempMultiply = multiply(qkt, ai);
            double v = -1 * tempMultiply->get(0, 0);
            Matrix* tmp = multiply(qk, v);
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
    return make_pair(Q, R);
}

Matrix* multiply_wrapper(Matrix* a, Matrix* b) {
    // Logic
    // 1. copy matrixes to device
    // 2. call multiply kernel
    // 3. copy results to host
    // 4. free allocated device memory

    Matrix* ab = new Matrix(a->n, b->m);

    // part 1 start
    Matrix *a_dev, *b_dev, *ab_dev;
    double *a_arr, *b_arr, *ab_arr;
    copyMatrixFromHostToDevice(a, &a_dev, &a_arr);
    copyMatrixFromHostToDevice(b, &b_dev, &b_arr);
    copyMatrixFromHostToDevice(ab, &ab_dev, &ab_arr);
    // part 1 end

    // part 2 start
    multiply<<<128, 32>>>(a_dev, b_dev, ab_dev);
    CCE(cudaGetLastError())
    // part 2 end 

    // part 3 start
    copyMatrixFromDeviceToHost(ab_arr, &ab, a->n, b->m);
    // part 3 end 


    // part 4 start
    CCE(cudaFree(a_arr));
    CCE(cudaFree(a_dev));
    CCE(cudaFree(b_arr));
    CCE(cudaFree(b_dev));
    CCE(cudaFree(ab_arr));
    CCE(cudaFree(ab_dev));
    // part 4 end
    return ab;
}

Triple* SVDDecomposition(Matrix *a, int rank, double eps) {
    int n = a->n;
    int m = a->m;
    auto u = randomMatrix(n, rank), sgm = randomMatrix(rank, rank), v = randomMatrix(m, rank);
    auto at = transpose(a);
    double err = 1e9;
    for (; err > eps;) {

        // auto av = multiply(a, v);
        auto av = multiply_wrapper(a, v);

        // show(av, a->n, rank);
        // show(av_test, a->n, rank);
        // exit(0);
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
        // av = multiply(a, v);
        auto usgm = multiply(u, sgm);
        double revert = -1;
        Matrix* usgmt = multiply(usgm, revert);
        auto difff = sum(av, usgmt);
        err = matrixNorm(difff);
        // double av_diff = diff(av, av_test);
        printf("Iteration ended, error=%f, diff=%f\n", err, 0.f);

        delete av;
        // delete av_test;
        delete qr_av.first;
        delete qr_av.second;
        delete atu;
        delete qr_atu.first;
        delete qr_atu.second;
        delete usgm;
        delete usgmt;
        delete difff;
    }
    delete at;
    return new Triple(u, sgm, v);
}