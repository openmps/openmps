#define MPS_GC
#define DIM3

#include <gtest/gtest.h>
#include "../Computer.hpp"

// detM != 0 case
TEST(MatrixTest, InvertMatrix){
	// 2次元テスト行列を用意
  //
	// 2成分のみ存在 その1
	// M            invM
	// [ 1, 0 ]     [ 1, 0 ]
	// [ 0, 2 ]     [ 0, 1/2]
	double Ms[] = {1.0,0.0,0.0,2.0};
	double iMs[] = {1.0,0.0,0.0,1.0/2.0};

	// 2成分のみ存在 その2
	// M            invM
	// [ 0, 2 ]     [  0, 1 ]
	// [ 1, 0 ]     [ 1/2, 0 ]
	double Ms2[] = {0.0,1.0,2.0,0.0};
	double iMs2[] = {0.0,1.0/2.0,1.0,0.0};

	// TODO: テスト自体を関数化
#ifdef DIM3 
	// 3次元に対しては、
  // [ 1  0  0   ]
  // [ 0  M0 M2 ]
  // [ 0  M1 M3 ] として2次元テスト行列を埋め込む
	auto M = OpenMps::Detail::CreateMatrix(1.0,0.0,0.0,0.0,Ms[0],Ms[1],0.0,Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( 1.0, invM(0,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(1,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(2,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(0,1) );
	ASSERT_DOUBLE_EQ( iMs[0], invM(1,1) );
	ASSERT_DOUBLE_EQ( iMs[1], invM(2,1) );
	ASSERT_DOUBLE_EQ( 0.0, invM(0,2) );
	ASSERT_DOUBLE_EQ( iMs[2], invM(1,2) );
	ASSERT_DOUBLE_EQ( iMs[3], invM(2,2) );

	M = OpenMps::Detail::CreateMatrix(1.0,0.0,0.0,0.0,Ms2[0],Ms2[1],0.0,Ms2[2],Ms2[3]);
	invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( 1.0, invM(0,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(1,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(2,0) );
	ASSERT_DOUBLE_EQ( 0.0, invM(0,1) );
	ASSERT_DOUBLE_EQ( iMs2[0], invM(1,1) );
	ASSERT_DOUBLE_EQ( iMs2[1], invM(2,1) );
	ASSERT_DOUBLE_EQ( 0.0, invM(0,2) );
	ASSERT_DOUBLE_EQ( iMs2[2], invM(1,2) );
	ASSERT_DOUBLE_EQ( iMs2[3], invM(2,2) );
#else

	auto M = OpenMps::Detail::CreateMatrix(Ms[1],Ms[2],Ms[3],Ms[4]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( iMs[1], invM(0,0) );
	ASSERT_DOUBLE_EQ( iMs[2], invM(0,1) );
	ASSERT_DOUBLE_EQ( iMs[3], invM(1,0) );
	ASSERT_DOUBLE_EQ( iMs[4], invM(1,1) );

	M = OpenMps::Detail::CreateMatrix(Ms2[1],Ms2[2],Ms2[3],Ms2[4]);
	invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	ASSERT_DOUBLE_EQ( iMs2[1], invM2(0,0) );
	ASSERT_DOUBLE_EQ( iMs2[2], invM2(0,1) );
	ASSERT_DOUBLE_EQ( iMs2[3], invM2(1,0) );
	ASSERT_DOUBLE_EQ( iMs2[4], invM2(1,1) );
#endif
}
