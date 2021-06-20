#include <iostream>
#include <chrono>
#include <lapacke.h>
#include "matrix_utils/operations.h"
#include "matrix_utils/svd.h"
#include "matrix/Matrix.h"

int main() {

    Matrix *m = hilbert(100, 100, 100, 10, 10);

    auto start = chrono::high_resolution_clock::now();
    auto tt = TTDecomposition(m, 1e-3);
    auto end   = chrono::high_resolution_clock::now();

    auto diff = chrono::duration_cast<chrono::milliseconds>(end - start);
    printf("Execution time %f", diff.count() / 1000.);

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
    return 0;
}
