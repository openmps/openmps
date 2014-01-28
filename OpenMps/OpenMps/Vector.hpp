#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED
#include "defines.hpp"

#include <boost/numeric/ublas/vector.hpp>

namespace OpenMps
{
	// 次元数
	const int DIM = DIMENSTION;

	// 2次元ベクトル
	typedef boost::numeric::ublas::c_vector<double, DIM> Vector;

	// 定数ベクトル
	class VectorConst : public Vector
	{
	public:
		VectorConst(const double& value)
		{
			std::fill_n(this->begin(), DIM, value);
		}
	};

	// ゼロベクトル
	const Vector VectorZero = VectorConst(0.0);
}
#endif