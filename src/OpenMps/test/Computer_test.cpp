#define MPS_GC

#include <gtest/gtest.h>
#include "../Computer.hpp"

// detM != 0 case
TEST(MatrixTest, InvertDetNotZero){
#ifdef DIM3
#else
	// 全成分入り, 対称性低そうな行列
	// M            invM
	// [ 1, 3 ]     [-2, 3/2]
	// [ 2, 4 ]     [ 1, -1/2]
	auto M = OpenMps::Detail::CreateMatrix(1.0,2.0,3.0,4.0);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( -2.0, invM(0,0) );
	ASSERT_DOUBLE_EQ( 1.0, invM(0,1) );
	ASSERT_DOUBLE_EQ( 3.0/2.0, invM(1,0) );
	ASSERT_DOUBLE_EQ( -1.0/2.0, invM(1,1) );

	// 2成分のみ存在 その1
	// M            invM
	// [ 1, 0 ]     [ 1, 0 ]
	// [ 0, 2 ]     [ 0, 1/2]
	M = OpenMps::Detail::CreateMatrix(1.0,0.0,0.0,2.0);
	invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( 1.0, invM(0,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(0,1) );
	ASSERT_DOUBLE_EQ( 0.0, invM(1,0) );
	ASSERT_DOUBLE_EQ( 1.0/2.0, invM(1,1) );

	// 2成分のみ存在 その2
	// M            invM
	// [ 0, 2 ]     [  0, 1 ]
	// [ 1, 0 ]     [ 1/2, 0 ]
	M = OpenMps::Detail::CreateMatrix(0.0,1.0,2.0,0.0);
	invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( 0.0, invM(0,0) );
	ASSERT_DOUBLE_EQ( 1.0/2.0, invM(0,1) );
	ASSERT_DOUBLE_EQ( 1.0, invM(1,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(1,1) );
#endif
}
