#define MPS_GC

#include <gtest/gtest.h>
#include "../Computer.hpp"

double dist_matrix(const double* M1, const double* M2){
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
	const double Ms[] = {1.0,0.0,0.0,2.0};
	const double iMs[] = {1.0,0.0,0.0,1.0/2.0};

	auto M = OpenMps::Detail::CreateMatrix(Ms[0],Ms[1],Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	const double M1[] = { iMs[0], iMs[1], iMs[2], iMs[3] };
	const double M2[] = { invM(0,0), invM(0,1), invM(1,0), invM(1,1) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
}

TEST(MatrixTest, InvertMatrixDiag2){
	// 2成分のみ存在 その2
	// M            invM
	// [ 0, 1 ]     [0, 1/2]
	// [ 2, 0 ]     [1, 0  ]
	const double Ms[] = {0.0,1.0,2.0,0.0};
	const double iMs[] = {0.0,1.0/2.0,1.0,0.0};

	auto M = OpenMps::Detail::CreateMatrix(Ms[0],Ms[1],Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	const double M1[] = { iMs[0], iMs[1], iMs[2], iMs[3] };
	const double M2[] = { invM(0,0), invM(0,1), invM(1,0), invM(1,1) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
}

TEST(MatrixTest, InvertMatrixAsym){
	// 非対称的
  // 全成分入り, 対称性低そうな行列
  // M            invM
  // [ 1, 2 ]     [-2,   1 ]
  // [ 3, 4 ]     [3/2,-1/2]
	const double Ms[] = {1.0,2.0,3.0,4.0};
	const double iMs[] = {-2.0,1.0,3./2.,-1./2.};

	auto M = OpenMps::Detail::CreateMatrix(Ms[0],Ms[1],Ms[2],Ms[3]);
	auto invM = OpenMps::Detail::InvertMatrix(std::move(M), 1.0);

	const double M1[] = { iMs[0], iMs[1], iMs[2], iMs[3] };
	const double M2[] = { invM(0,0), invM(0,1), invM(1,0), invM(1,1) };
	ASSERT_DOUBLE_EQ( dist_matrix(M1,M2), 0.0 );
}
