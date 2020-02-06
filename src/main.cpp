#include <iostream>
#include "matrix_utils/operations.h"
#include "matrix_utils/svd.h"
#include "matrix/Matrix.h"

int main() {
    Matrix* m = hilbert(2, 2);
//    Matrix* m2 = hilbert(2, 2);
//    auto qr = QRDecompositionNaive(m);
//    Matrix* mt = multiply(qr.first, qr.second);
//    cout << diff(m, mt);
//    m->show();
//    m->set(2, 1, 5);
//    m->show();
//    cout << diff(m, mt);
    m->show();
//    mt->show();
//    sum(m, m2)->show();
//    multiply(m, 2)->show();
//    subMatrix(m, 0, 2, 0, 2)->show();
//    transpose(m)->show();
    return 0;
}
