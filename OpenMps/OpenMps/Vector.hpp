#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED

#pragma warning(push, 0)
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/numeric/ublas/vector.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#pragma warning(pop)

#include "defines.hpp"

namespace OpenMps
{
	// 2次元ベクトル
	typedef boost::numeric::ublas::c_vector<double, DIM> Vector;

	// 2次元ベクトルを作成する
	static inline Vector CreateVector(const double v1, const double v2)
	{
		Vector vec;
		vec[0] = v1;
		vec[1] = v2;
		return vec;
	}

	// ゼロベクトル
	static const Vector VectorZero(CreateVector(0, 0));
}
#endif
