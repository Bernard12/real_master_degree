#include <iostream>
#include <chrono>
#include <lapacke.h>
#include "matrix_utils/operations.h"
#include "matrix_utils/svd.h"
#include "matrix/Matrix.h"

int main() {
//    Matrix *m = hilbert(100, 100, 100, 10, 10);

//    auto start = chrono::high_resolution_clock::now();
//    auto tt = TTDecomposition(m, 1e-3);
//    auto end   = chrono::high_resolution_clock::now();

//    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
//    printf("Execution time %f", diff.count() / 1000.);

//    Matrix *m2 = m->copy();
//    m->show();
//    vector<int> bla = {2, 3};
//    m->reshape(bla);
//    m->show();
//    Matrix *m = new Matrix(2, 2);
//    m->set(0, 0, 1);
//    m->set(0, 1, 2);
//    m->set(1, 0, 2);
//    m->set(1, 1, 4);
////    m->show();
//    auto bla = SVDDecomposition(m);
//    bla->first->show();
//    bla->second->show();
//    bla->third->show();
//    Matrix *temp = multiply(bla->first, bla->second);
//    Matrix *temp2 = multiply(temp, bla->third);
//    temp2->show();
//    delete m;
//    delete m2;
//    tt[0]->show();
//    tt[1]->get2DshapeFrom3d(0)->show();
//    tt[1]->get2DshapeFrom3d(1)->show();
//    tt[1]->get2DshapeFrom3d(2)->show();
//    tt[2]->show();
//    tt[0]->show();
//    vector<int> indexes = { 1, 1, 1 };
//    printf("%f", getValueFromTrain(tt, indexes));
//    for (int i = 0; i < 3; i++) {
//        delete tt[i];
//    }
//    delete m;
//    double res = 0;
//    int r = 5;
//    double step = 1. / r;
//
//    printf("Step: %f; Razbienie: %d\n", step, r);
//    auto start = chrono::high_resolution_clock::now();
//    for (int i1 = 0; i1 < r; i1++) {
//        for (int i2 = 0; i2 < r; i2++) {
//            for (int i3 = 0; i3 < r; i3++) {
//                for (int i4 = 0; i4 < r; i4++) {
//                    for (int i5 = 0; i5 < r; i5++) {
//                        for (int i6 = 0; i6 < r; i6++) {
//                            for (int i7 = 0; i7 < r; i7++) {
//                                for (int i8 = 0; i8 < r; i8++) {
//                                    for (int i9 = 0; i9 < r; i9++) {
//                                        for (int i10 = 0; i10 < r; i10++) {
//                                            double sum =
//                                                    i1 * step +
//                                                    i2 * step +
//                                                    i3 * step +
//                                                    i4 * step +
//                                                    i5 * step +
//                                                    i6 * step +
//                                                    i7 * step +
//                                                    i8 * step +
//                                                    i9 * step +
//                                                    i10 * step;
//                                            double sn = sin(sum);
//                                            res += sn * (step * step * step * step * step * step * step * step * step * step);
//                                        }
//                                    }
//                                }
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    printf("%.6f\n", res);
//    auto end   = chrono::high_resolution_clock::now();
//
//    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
//    printf("Execution time %f", diff.count() / 1000.);
    double res = 0;
    int r = 7;
    double step = 1. / r;

    auto cube = sinCube(r, step);


    auto start = chrono::high_resolution_clock::now();
    auto tt = TTDecomposition(cube, 1e-3);
    vector<Matrix *> u;

    for (int i = 0; i < 10; i++) {
        auto us = new Matrix(r, 1);
        for (int j = 0; j < r; j++) {
            us->set(j, 0, step);
        }
        u.push_back(us);
    }

    res = convolution(tt, u);
    auto end   = chrono::high_resolution_clock::now();


    printf("Res: %.6f\n", res);

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("Execution time %f", diff.count() / 1000.);


    delete cube;
    return 0;
}
