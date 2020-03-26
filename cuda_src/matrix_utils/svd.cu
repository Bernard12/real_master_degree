#include "./svd.cuh"
#include <cublas_v2.h>
#include <cusolverDn.h>
#define STEP(i) printf("Step %d\n", i);


#define CCE(errValue)                                                   \
    do {                                                                \
        if (errValue != cudaSuccess) {                                  \
            fprintf(stderr ,"[CUDA-ERROR]-[%s(line:%d)]: %s\n", __FILE__, __LINE__, cudaGetErrorString(errValue)); \
            exit(0);                                                    \
        }                                                               \
    } while(0);

// CATCH_CUDA_ERR(cudaMalloc(&dev_array, sizeof(int) * used_n));
// CATCH_CUDA_ERR(cudaMemcpy(dev_array, array, sizeof(int) * used_n, cudaMemcpyHostToDevice));

void copyMatrixFromHostToDevice(Matrix* hostMatrix, Matrix** deviceMatrix, double** deviceMatrixArray, int** deviceDimsArray) {
    const int n = hostMatrix->n(), m = hostMatrix->m();
    Matrix* temp = new Matrix(n, m);
    delete[] temp->matrix;
    delete[] temp->real_shape;

    const int matrix_size = sizeof(double) * n * m;
    CCE(cudaMalloc(&temp->matrix, matrix_size));
    CCE(cudaMemcpy(temp->matrix, hostMatrix->matrix, matrix_size, cudaMemcpyHostToDevice));

    const int dims_size = sizeof(int) * 2;
    CCE(cudaMalloc(&temp->real_shape, dims_size));
    CCE(cudaMemcpy(temp->real_shape, hostMatrix->real_shape, dims_size, cudaMemcpyHostToDevice));

    CCE(cudaMalloc(deviceMatrix, sizeof(Matrix) * 1));
    CCE(cudaMemcpy(*deviceMatrix, temp, sizeof(Matrix) * 1, cudaMemcpyHostToDevice));

    *deviceMatrixArray = temp->matrix;
    *deviceDimsArray = temp->real_shape;
    temp->matrix = NULL;
    temp->real_shape = NULL;
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
    int n = a->n();
    int m = a->m();
    Matrix* Q = new Matrix(n, m);
    Matrix* R = new Matrix(m, m);
    for (int i = 0; i < m; i++) {
        Matrix* ai = subMatrix(a, 0, n + 0, i + 0, i + 1);
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

    Matrix* ab = new Matrix(a->n(), b->m());

    // part 1 start
    Matrix *a_dev, *b_dev, *ab_dev;
    double *a_arr, *b_arr, *ab_arr;
    int *a_dim_arr, *b_dim_arr, *ab_dim_arr;
    copyMatrixFromHostToDevice(a, &a_dev, &a_arr, &a_dim_arr);
    copyMatrixFromHostToDevice(b, &b_dev, &b_arr, &b_dim_arr);
    copyMatrixFromHostToDevice(ab, &ab_dev, &ab_arr, &ab_dim_arr);
    delete ab;
    // part 1 end

    // part 2 start
    // cudaError_t cudaStat; // cudaMalloc status
    // cublasHandle_t handle; 
    // cublasStatus_t stat = cublasCreate(&handle); // CUBLAS functions status
    // double alpha = 1.0, beta = 0.0;
    // cublasDgemm(handle,
    //     CUBLAS_OP_N, CUBLAS_OP_N,
    //     a->n, b->m, b->n,
    //     &alpha,
    //     a_arr, a->n,
    //     b_arr, b->n,
    //     &beta,
    //     ab_arr, a->n
    // );
    // CCE(cudaDeviceSynchronize());
    // CCE(cudaGetLastError())
    
    multiply<<<128, 32>>>(a_dev, b_dev, ab_dev);
    CCE(cudaGetLastError())
    // part 2 end 

    // part 3 start
    copyMatrixFromDeviceToHost(ab_arr, &ab, a->n(), b->m());
    // part 3 end 


    // part 4 start
    CCE(cudaFree(a_arr));
    CCE(cudaFree(a_dev));
    CCE(cudaFree(b_arr));
    CCE(cudaFree(b_dev));
    CCE(cudaFree(ab_arr));
    CCE(cudaFree(ab_dev));

    CCE(cudaFree(a_dim_arr));
    CCE(cudaFree(b_dim_arr));
    CCE(cudaFree(ab_dim_arr));
    // part 4 end
    return ab;
}

Triple* SVDDecomposition(Matrix *a, int rank, double eps) {
    int n = a->n();
    int m = a->m();
    auto u = randomMatrix(n, rank), sgm = randomMatrix(rank, rank), v = randomMatrix(m, rank);
    auto at = transpose(a);
    double err = 1e9;
    for (; err > eps;) {

        Matrix* av = multiply_wrapper(a, v);
        // auto av = multiply(a, v);

        // show(av, a->n, rank);
        // show(av_test, a->n, rank);
        // exit(0);
        pair<Matrix*, Matrix*> qr_av = QRDecompositionNaive(av);

        Matrix* u_tmp = subMatrix(qr_av.first, 0, n, 0, rank);
        delete u;
        u = u_tmp;

        Matrix* atu = multiply_wrapper(at, u);
        // auto atu = multiply(at, u);
        pair<Matrix*, Matrix*> qr_atu = QRDecompositionNaive(atu);

        Matrix* v_tmp = subMatrix(qr_atu.first, 0, m, 0, rank);
        delete v;
        v = v_tmp;

        Matrix* sgm_tmp = subMatrix(qr_atu.second, 0, rank, 0, rank);
        delete sgm;
        sgm = sgm_tmp;

        // find error e = || A*V - U*SGM||
        // av = multiply(a, v);
        auto usgm = multiply_wrapper(u, sgm);
        // auto usgm = multiply(u, sgm);
        double revert = -1;
        Matrix* usgmt = multiply(usgm, revert);
        Matrix* difff = sum(av, usgmt);
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

// Only for rectangal matrix
Triple* SVDDecompositionwCUB(Matrix *t) {
    if (t->shape_length!= 2) {
        printf("Cannot perform SVD for non rectangular matrix!");
        exit(-1);
    }
    cusolverDnHandle_t cusolverH; // cusolver handle
    cusolverStatus_t cusolvstatus = cusolverDnCreate(&cusolverH);
    printf("CUSOLVE %d\n", cusolvstatus);

    Matrix* a;

    if (t->n() < t->m()) {
        a = transpose(t);
    } else {
        a = t->copy();
    }

    int rows = a->n(), cols = a->m();
    int rank = min(rows, cols);

    double* a_arr;
    CCE(cudaMalloc(&a_arr, sizeof(double) * rows * cols));
    CCE(cudaMemcpy(a_arr, a->matrix, sizeof(double) * rows * cols, cudaMemcpyHostToDevice));

    // Matrix U SIGMA V on host
    double *U_arr, *VT_arr;
    CCE(cudaMalloc(&U_arr, sizeof(double) * rows * rank));
    CCE(cudaMalloc(&VT_arr, sizeof(double) * rank * cols));

    // array for singular values
    double* s_arr;
    CCE(cudaMalloc(&s_arr, sizeof(double) * rank));
    CCE(cudaGetLastError());

    int lda = rows;
    int ldu = rows;
    int ldvt = cols;

    int lwork = 0;
    auto status = cusolverDnDgesvd_bufferSize(cusolverH, rows, cols, &lwork);
    printf("Buff size status %d\n", status);
    CCE(cudaGetLastError());
    double* work;
    CCE(cudaMalloc(&work, sizeof(double) * lwork));

    double* d_rwork;
    CCE(cudaMalloc(&d_rwork, sizeof(double) * (rank - 1)));
    
    int* info;
    CCE(cudaMalloc(&info, sizeof(int)));
    cusolverStatus_t cusolver_status;
    cusolver_status = cusolverDnDgesvd(
        cusolverH,
        'S', 'S',
        rows, cols,
        a_arr, lda,
        s_arr,
        U_arr, ldu,
        VT_arr, ldvt,
        work, lwork,
        NULL, info
    );

    CCE(cudaGetLastError());
    printf("Debug: rows=%d cols=%d, lda=%d ldu=%d ldvt=%d, lwork=%d, \n", rows, cols, lda, ldu, ldvt, lwork);
    printf("Checks lda!<max(1,rows): %d\n", !(lda < max(1, rows)));
    printf("Checks ldu!<max(1,rows): %d\n", !(ldu < max(1, rows)));
    printf("Checks ldv!<max(1,cols): %d\n", !(ldvt < max(1, cols)));
    printf("cuBLAS SVD result status: %d\n", cusolver_status);
    // printf("%d %d %d %d\n", CUSOLVER_STATUS_SUCCESS, CUSOLVER_STATUS_NOT_INITIALIZED, CUSOLVER_STATUS_INVALID_VALUE, CUSOLVER_STATUS_ARCH_MISMATCH, CUSOLVER_STATUS_INTERNAL_ERROR);

    double* s_cpu = new double[rank];
    CCE(cudaMemcpy(s_cpu, s_arr, sizeof(double) * rank, cudaMemcpyDeviceToHost));


    Matrix* S = new Matrix(rank, rank);
    for (int i = 0; i < rank; i++) {
        S->set(i, i, s_cpu[i]);
    }
    delete[] s_cpu;

    Matrix* U = new Matrix(rows, rank);
    CCE(cudaMemcpy(U->matrix, U_arr, sizeof(double) * rows * rank, cudaMemcpyDeviceToHost));

    Matrix* VT = new Matrix(rank, cols);
    CCE(cudaMemcpy(VT->matrix, VT_arr, sizeof(double) * rank * cols, cudaMemcpyDeviceToHost));

    if (t->n() < t->m()) {
        auto real_U = transpose(VT);
        auto real_VT = transpose(U);

        return new Triple(real_U, S, real_VT);
    } else {
        return new Triple(U, S, VT);
    }
}

vector<Matrix*> TTDecomposition(Matrix* a, double eps) {
    vector<Matrix *> res;

    double norm = frobeniousNorm(a);
    double threshold = norm * eps / sqrt(a->shape_length - 1);

    int n_left = a->real_shape[0];
    int n_right = 1;
    for (int i = 1; i < a->shape_length; i++) {
        n_right *= a->real_shape[i];
    }

    int shape[] = { n_left, n_right };
    Matrix *M = a->copy();
    M->reshape(shape, 2);

    // U S VT
    auto svd_m_full = SVDDecompositionwCUB(M);
    auto svd_m = trunkSVDResultsForTT(svd_m_full, eps);
    res.push_back(svd_m->first);
    int r = svd_m->second->real_shape[0];
    delete M;
    M = multiply(svd_m->second, svd_m->third);

    delete svd_m_full->first;
    delete svd_m_full->second;
    delete svd_m_full->third;
    delete svd_m_full;
    delete svd_m->second;
    delete svd_m->third;
    delete svd_m;


    for (int i = 1; i < a->shape_length - 1; i++) {
        n_left = a->real_shape[i];
        n_right = n_right / a->real_shape[i];

        int next_shape[] = {r * n_left, n_right};
        M->reshape(next_shape, 2);

        auto svd_m_next_full = SVDDecompositionwCUB(M);
        auto svd_m_next = trunkSVDResultsForTT(svd_m_next_full, eps);
        int r_cur = svd_m_next->second->real_shape[0];

        Matrix *GK = svd_m_next->first;
        int gk_shape[] = {r, n_left, r_cur};
        GK->reshape(gk_shape, 3);

        res.push_back(GK);
        r = r_cur;

        delete M;
        M = multiply(svd_m_next->second, svd_m_next->third);

        delete svd_m_next_full->first;
        delete svd_m_next_full->second;
        delete svd_m_next_full->third;
        delete svd_m_next_full;
        delete svd_m_next->second;
        delete svd_m_next->third;
        delete svd_m_next;
    }

    res.push_back(M);
    return res;
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

double getValueFromTrain(vector<Matrix *> m, vector<int> indexes) {
    Matrix *first = subMatrix(m[0], indexes[0], indexes[0] + 1, 0, m[0]->m());
    for (int i = 1; i < m.size() - 1; i++) {
        Matrix *cur = m[i]->get2DshapeFrom3d(indexes[i]);

        auto temp_first = multiply(first, cur);
        delete first;
        first = temp_first;

        delete cur;
    }
    Matrix *last = subMatrix(m[m.size() - 1], 0, m[m.size() - 1]->n(), indexes[indexes.size() - 1],
                             indexes[indexes.size() - 1] + 1);

    auto temp_first = multiply(first, last);

    double res = temp_first->matrix[0];

    delete first;
    delete last;
    delete temp_first;
    return res;
}