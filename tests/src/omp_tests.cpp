//
// Created by ivan on 26.01.2020.
//

#include <vector>
#include <iostream>
#include <omp.h>
#include "gtest/gtest.h"

using namespace std;

TEST(simple, test) {
    ASSERT_EQ(1, 1);
}

TEST(simple, test_no_openmp) {
    const int N = 1024 * 1024 * 1024;
    vector<char> vec(N, 0);
    for (int i = 0; i < N; i++) {
        vec[i] += 10;
        vec[i] *= 3.1415;
        vec[i] -= 12;
    }
}

TEST(simple, test_openmp) {
    const int N = 1024 * 1024 * 1024;
    vector<char> vec(N, 0);
#pragma omp parallel for shared(vec)
    for (int i = 0; i < N; i++) {
        vec[i] += 10;
        vec[i] *= 3.1415;
        vec[i] -= 12;
    }
}
