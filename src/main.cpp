#include <iostream>
#include "matrix_utils/operations.h"
#include "matrix_utils/svd.h"
#include "matrix/Matrix.h"

int main() {
//    Matrix mtr(4, 3);
//    mtr[0][0] = 1;
//    mtr[0][1] = -1;
//    mtr[0][2] = 4;
//    mtr[1][0] = 1;
//    mtr[1][1] = 4;
//    mtr[1][2] = -2;
//    mtr[2][0] = 1;
//    mtr[2][1] = 4;
//    mtr[2][2] = 2;
//    mtr[3][0] = 1;
//    mtr[3][1] = -1;
//    mtr[3][2] = 0;
    Matrix mtr(3, 3);
    mtr[0][0] = 1;
    mtr[0][1] = 1;
    mtr[0][2] = 0;

    mtr[1][0] = 1;
    mtr[1][1] = 0;
    mtr[1][2] = 1;

    mtr[2][0] = 0;
    mtr[2][1] = 1;
    mtr[2][2] = 1;

    QRDecompositionNaive(mtr);
    return 0;
}
