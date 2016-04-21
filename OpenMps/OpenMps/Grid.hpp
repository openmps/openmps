#ifndef GRID_INCLUDED
#define GRID_INCLUDED

#pragma warning(push, 0)
#include <boost/multi_array.hpp>
#pragma warning(pop)

#include "Vector.hpp"

namespace OpenMps
{
	// 近傍粒子探索用グリッド
	class Grid final
	{
	private:
		// 1ブロックの長さ（影響半径に等しい）
		const double blockLength;

		// 原点（計算空間の最大座標に等しい）
		const Vector origin;

		// 各ブロックの粒子番号
		boost::multi_array<std::size_t, DIM + 1> data;

		using Index = decltype(data)::index;

		static auto Ceil(const double a, const double b)
		{
			return static_cast<Index>(std::ceil(a / b));
		}

		// 水平方向の最大ブロック数
		auto GridSizeX() const
		{
			return static_cast<Index>(data.shape()[0]);
		}
		// 鉛直方向の最大ブロック数
		auto GridSizeZ() const
		{
			return static_cast<Index>(data.shape()[1]);
		}
		// 1ブロック内の最大粒子数
		auto MaxParticles() const
		{
			return static_cast<Index>(data.shape()[2]) - 1; // 先頭の粒子数の格納分を除く
		}

		// 各ブロック内の粒子数
		auto& ParticleCount(const Index i, const Index j)
		{
			return data[i][j][0]; // 先頭は粒子数を格納してある
		}
		auto ParticleCount(const Index i, const Index j) const
		{
			return data[i][j][0]; // 先頭は粒子数を格納してある
		}

		// 各ブロック内の粒子
		auto& Particle(const Index i, const Index j, const Index k)
		{
			return data[i][j][k + 1]; // 先頭は粒子数を格納してある
		}
		auto Particle(const Index i, const Index j, const Index k) const
		{
			return data[i][j][k + 1]; // 先頭は粒子数を格納してある
		}

	public:

		// @param r_e 影響半径
		// @param l_0 初期粒子間距離
		// @param minX 計算空間の最小座標
		// @param maxX 計算空間の最大座標
		Grid(const double r_e, const double l_0,
			const Vector& minX, const Vector& maxX)
			: blockLength(r_e), origin(minX),
			data(boost::extents
				[Ceil(maxX[0] - minX[0], r_e)+2] // 水平方向の最大ブロック数（少し大きめにとっておく）
				[Ceil(maxX[1] - minX[1], r_e)+2] // 鉛直方向の最大ブロック数（少し大きめにとっておく）
				[1 + (Ceil(r_e, l_0)+1)*(Ceil(r_e, l_0) + 1)]) // 1ブロック内の最大粒子数＋存在する粒子数
		{}

		Grid(const Grid&) = delete;
		Grid(Grid&&) = delete;
		Grid& operator=(const Grid&) = delete;

		// 全消去（全ブロックの粒子数を0にする）
		void Clear()
		{
			const auto n = GridSizeX();
			const auto m = GridSizeZ();
			for(auto i = decltype(n)(0); i < n; i++)
			{
				for(auto j = decltype(m)(0); j < m; j++)
				{
					ParticleCount(i, j) = 0;
				}
			}
		}

		struct Exception : public std::runtime_error
		{
		public:
			Exception(std::string&& msg)
				: std::runtime_error(msg)
			{}
		};

		// 粒子を格納する
		bool Store(const Vector& x, const std::size_t index)
		{
			const auto i = Ceil(x[0] - origin[0], blockLength);
			const auto j = Ceil(x[1] - origin[1], blockLength);
			const auto n = GridSizeX();
			const auto m = GridSizeZ();

			// 領域外なら格納しない
			if(
				(0 <= i) && (i < n) &&
				(0 <= j) && (j < m))
			{
				const auto k = static_cast<Index>(ParticleCount(i, j));
				const auto maxCount = MaxParticles();
				if(k > maxCount)
				{
					throw Exception("Too many particle in a block");
				}
				Particle(i, j, k) = index;

				return true;
			}
			else
			{
				return false;
			}
		}
	};
}
#endif
