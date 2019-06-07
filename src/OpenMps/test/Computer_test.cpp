#include <gtest/gtest.h>

#define MPS_GC
#include "../Computer.hpp"

TEST(matrix_test, invert_test){
#ifdef DIM3
#else // two-dimensional case
  // M            invM
  // [ 1, 3 ]     [-2, 3/2]
  // [ 2, 4 ]     [ 1, -1/2]

  // 行列添字 (0,0), (0,1), (1,0), (1,1) という順番
  auto M = OpenMps::Detail::CreateMatrix(1.0,2.0,3.0,4.0);
  auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

  ASSERT_EQ( -2.0, invM(0,0) );
  ASSERT_EQ( 1.0, invM(0,1) );
  ASSERT_EQ( 3.0/2.0, invM(1,0) );
  ASSERT_EQ( -1.0/2.0, invM(1,1) );
#endif
}
