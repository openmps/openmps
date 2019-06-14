#define MPS_GC
#define DIM3

#include <gtest/gtest.h>
#include "../Computer.hpp"

double dist_matrix(double* M1, double* M2){
	double d = 0.0;
	double dim = OpenMps::DIM;

	for(int k = 0; k < dim*dim; ++k){
		double diff = M1[k] - M2[k];
		d += diff*diff;
	}
	return sqrt(d);
}

TEST(MatrixTest, InvertMatrixDiag){
	// 2成分のみ存在するケース
	// M            invM
	// [ 1, 0 ]     [ 1, 0 ]
	// [ 0, 2 ]     [ 0, 1/2]
	double Ms[] = {1.0,0.0,0.0,2.0};
	double iMs[] = {1.0,0.0,0.0,1.0/2.0};

#ifdef DIM3
	// 3次元に対しては、
	// [ 1  0  0   ]
	// [ 0  M0 M2 ]
	// [ 0  M1 M3 ] として2次元テスト行列を埋め込む
	auto M = OpenMps::Detail::CreateMatrix(1.0,0.0,0.0,0.0,Ms[0],Ms[1],0.0,Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	// M1: 用意しておいた答え, M2: InvertMatrix計算結果
	double M1[] = { 1.0, 0.0, 0.0, 0.0, iMs[0], iMs[1], 0.0, iMs[2], iMs[3] };
	double M2[] = { invM(0,0), invM(1,0), invM(2,0), invM(0,1), invM(1,1), invM(2,1), invM(0,2), invM(1,2), invM(2,2) };
	// 成分ごとの差を足したdist_matrixが0.0と等しいか？
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
#else
	auto M = OpenMps::Detail::CreateMatrix(Ms[0],Ms[1],Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	double M1[] = { iMs[0], iMs[1], iMs[2], iMs[3] };
	double M2[] = { invM(0,0), invM(0,1), invM(1,0), invM(1,1) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
#endif
}

TEST(MatrixTest, InvertMatrixDiag2){
	// 2成分のみ存在 その2
	// M            invM
	// [ 0, 2 ]     [  0, 1 ]
	// [ 1, 0 ]     [ 1/2, 0 ]
	double Ms[] = {0.0,1.0,2.0,0.0};
	double iMs[] = {0.0,1.0/2.0,1.0,0.0};

#ifdef DIM3
	auto M = OpenMps::Detail::CreateMatrix(1.0,0.0,0.0,0.0,Ms[0],Ms[1],0.0,Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	double M1[] = { 1.0, 0.0, 0.0, 0.0, iMs[0], iMs[1], 0.0, iMs[2], iMs[3] };
	double M2[] = { invM(0,0), invM(1,0), invM(2,0), invM(0,1), invM(1,1), invM(2,1), invM(0,2), invM(1,2), invM(2,2) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
#else
	auto M = OpenMps::Detail::CreateMatrix(Ms[0],Ms[1],Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	double M1[] = { iMs[0], iMs[1], iMs[2], iMs[3] };
	double M2[] = { invM(0,0), invM(0,1), invM(1,0), invM(1,1) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
#endif
}

TEST(MatrixTest, InvertMatrixAsym){
	// 非対称的
  // 全成分入り, 対称性低そうな行列
  // M            invM
  // [ 1, 3 ]     [-2, 3/2]
  // [ 2, 4 ]     [ 1, -1/2]
	double Ms[] = {1.0,2.0,3.0,4.0};
	double iMs[] = {-2.0,1.0,3./2.,-1./2.};

#ifdef DIM3
	auto M = OpenMps::Detail::CreateMatrix(1.0,0.0,0.0,0.0,Ms[0],Ms[1],0.0,Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	double M1[] = { 1.0, 0.0, 0.0, 0.0, iMs[0], iMs[1], 0.0, iMs[2], iMs[3] };
	double M2[] = { invM(0,0), invM(1,0), invM(2,0), invM(0,1), invM(1,1), invM(2,1), invM(0,2), invM(1,2), invM(2,2) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
#else
	auto M = OpenMps::Detail::CreateMatrix(Ms[0],Ms[1],Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	double M1[] = { iMs[0], iMs[1], iMs[2], iMs[3] };
	double M2[] = { invM(0,0), invM(0,1), invM(1,0), invM(1,1) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
#endif
}
