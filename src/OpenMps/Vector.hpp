#ifndef VECTOR_INCLUDED
#define VECTOR_INCLUDED

#pragma warning(push, 0)
#pragma warning(disable : 4996)
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

namespace { namespace OpenMps
{
	// ベクトル
	using Vector = boost::numeric::ublas::c_vector<double, DIM>;

	namespace Detail
	{
		template<decltype(DIM) D>
		struct CreateVector;

		template<>
		struct CreateVector<2>
		{
			static auto Get(const std::tuple<double, double>& val)
			{
				Vector vec;
				vec[0] = std::get<0>(val);
				vec[1] = std::get<1>(val);
				return vec;
			}

			static auto Get(const double val)
			{
				return Get(std::make_tuple(val, val));
			}
		};

		template<>
		struct CreateVector<3>
		{
			static auto Get(const std::tuple<double, double, double>& val)
			{
				Vector vec;
				vec[0] = std::get<0>(val);
				vec[1] = std::get<1>(val);
				vec[2] = std::get<2>(val);
				return vec;
			}

			static auto Get(const double val)
			{
				return Get(std::make_tuple(val, val, val));
			}
		};
	}

	// ベクトルを作成する
	template<typename T, typename... ARGS>
	inline auto CreateVector(const T val, const ARGS... args)
	{
		return Detail::CreateVector<DIM>::Get(std::make_tuple(val, args...));
	}
	template<typename T>
	inline auto CreateVector(const T val)
	{
		return Detail::CreateVector<DIM>::Get(val);
	}

	// ゼロベクトル
	static const auto VectorZero = CreateVector(0);
}}
#endif
