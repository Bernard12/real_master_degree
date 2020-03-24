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
Triple* SVDDecompositionwCUB(Matrix *a) {
    if (a->shape_length!= 2) {
        printf("Cannot perform SVD for non rectangular matrix!");
        exit(-1);
    }
    cusolverDnHandle_t cusolverH; // cusolver handle
    cusolverStatus_t cusolvstatus = cusolverDnCreate(&cusolverH);
    // printf("CUSOLVE %d\n", cusolvstatus);

    int n = a->n(), m = a->m();
    int rank = min(n, m);

    double* a_arr;
    CCE(cudaMalloc(&a_arr, sizeof(double) * n * m));
    CCE(cudaMemcpy(a_arr, a->matrix, sizeof(double) * n * m, cudaMemcpyHostToDevice));

    // Matrix U SIGMA V on host
    double *U_arr, *VT_arr;
    CCE(cudaMalloc(&U_arr, sizeof(double) * n * rank));
    CCE(cudaMalloc(&VT_arr, sizeof(double) * rank * m));

    // array for singular values
    double* s_arr;
    CCE(cudaMalloc(&s_arr, sizeof(double) * rank));
    CCE(cudaGetLastError());

    int lda = n;
    int ldu = n;
    int ldvt = rank;

    int lwork = 0;
    cusolverDnDgesvd_bufferSize(cusolverH, n, rank, &lwork);
    CCE(cudaGetLastError());
    double* work;
    CCE(cudaMalloc(&work, sizeof(double) * lwork));

    double* d_rwork;
    CCE(cudaMalloc(&d_rwork, sizeof(double) * (rank - 1)));
    
    int* info;
    CCE(cudaMalloc(&info, sizeof(int)));

    cusolverStatus_t cusolver_status = cusolverDnDgesvd(
        cusolverH,
        'S', 'S',
        n, m,
        a_arr, lda,
        s_arr,
        U_arr, ldu,
        VT_arr, ldvt,
        work, lwork,
        d_rwork, info
    );
    CCE(cudaGetLastError());
    // printf("cuBLAS SVD result status: %d\n", cusolver_status == CUSOLVER_STATUS_SUCCESS);

    double* s_cpu = new double[rank];
    CCE(cudaMemcpy(s_cpu, s_arr, sizeof(double) * rank, cudaMemcpyDeviceToHost));

    Matrix* U = new Matrix(n, rank);
    CCE(cudaMemcpy(U->matrix, U_arr, sizeof(double) * n * rank, cudaMemcpyDeviceToHost));

    Matrix* S = new Matrix(rank, rank);
    for (int i = 0; i < rank; i++) {
        S->set(i, i, s_cpu[i]);
    }

    Matrix* VT = new Matrix(rank, m);
    CCE(cudaMemcpy(VT->matrix, VT_arr, sizeof(double) * rank * m, cudaMemcpyDeviceToHost));

    return new Triple(U, S, VT);
}

vector<Matrix*> tensorTrain(Matrix* t, double eps) {
    // initialization 
    vector<Matrix*> res;
    
    // step 1
    STEP(1)
    double nrm = frobeniousNorm(t);
    // step 2
    STEP(2)
    int n_left = t->real_shape[0];
    int n_right = 1;
    for (int i = 1; i < t->shape_length; i++) {
        n_right *= t->real_shape[i];
    }
    // step 3 
    STEP(3)
    Matrix* B = t->copy();
    // step 4
    STEP(4)
    int shapes[2] = { n_left, n_right };
    B->reshape(shapes, 2);
    // show(B, n_left, n_right);
    // delete[] shapes;
    // step 5.1
    STEP(5)
    int rank = min(n_left, n_right);
    // TODO: use cublas SVD
    auto B_svd = SVDDecomposition(B, rank, 1e-6);
    show(B_svd->first, n_left, rank);
    CCE(cudaGetLastError())
    // step 5.2
    // TODO: need to find optimal way to approximate matrix rank!
    // double threshold = eps * nrm / sqrt(t->dims_count - 1);
    // double sigma_sum = 0;
    // int r = rank;
    // for (int i = rank - 1; i >= 0; i--) {
    //     double sigma_i = B_svd->second->get(i, i);
    //     double sigma_i_s = sigma_i * sigma_i;
    //     if (sigma_sum + sigma_i_s > threshold) {
    //         r = i + 1;
    //         break;
    //     } else {
    //         sigma_sum += sigma_i_s;
    //     }
    // }
    // step 6
    STEP(6)
    Matrix* G_1 = B_svd->first;
    res.push_back(G_1);
    delete B;
    B = multiply(B_svd->second, transpose(B_svd->third));
    // TODO: free B_svd memory
    int r_cur = rank;
    // Other G calc
    // step 8
    STEP(8)
    for (int i = 1; i < t->shape_length - 1; i++) {
        // step 9
        STEP(9)
        n_left = t->real_shape[i];
        n_right /= t->real_shape[i];

        // step 10
        STEP(10)
        int shapes[] = { r_cur * n_left, n_right };
        B->reshape(shapes, 2);

        // step 11
        STEP(11)
        int b_rank = min(r_cur * n_left, n_right );
        auto b_svd = SVDDecomposition(B, b_rank, 1e-6);

        // step 12
        STEP(12)
        int G_I_shapes[] = { r_cur, t->real_shape[i], b_rank };
        b_svd->first->reshape(G_I_shapes, 3);
        Matrix* G_I = b_svd->first;
        res.push_back(G_I);

        // step 13
        STEP(13)
        delete B;
        B = multiply(b_svd->second, transpose(b_svd->third));

        // Missing step 13.5
        r_cur = b_rank;
    }
    STEP(14)
    Matrix* G_D = B->copy();
    res.push_back(G_D);
    return res;
}