#ifndef GRID_INCLUDED
#define GRID_INCLUDED

#pragma warning(push, 0)
#include <iterator>
#include <boost/multi_array.hpp>
#pragma warning(pop)

#include "Vector.hpp"

namespace OpenMps
{
	// 近傍粒子探索用グリッド
	class Grid final
	{
	public:
		using ParticleID = std::size_t;

		// 探索対象ブロックの総数
		static constexpr std::size_t MAX_NEIGHBOR_BLOCK = 3 * 3; // 2次元なので

	private:
		// 1ブロックの長さ（影響半径に等しい）
		const double blockLength;

		// 原点（計算空間の最大座標に等しい）
		const Vector origin;

		// 各ブロックの粒子番号
		boost::multi_array<ParticleID, DIM + 1> data;

		using Index = decltype(data)::index;

		static auto Ceil(const double a, const double b)
		{
			return static_cast<Index>(std::ceil(a / b));
		}
		static auto Floor(const double a, const double b)
		{
			return static_cast<Index>(std::floor(a / b));
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

		// 各ブロック内の粒子数
		auto& ParticleCount(const Index i, const Index j)
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

		struct Exception : public std::runtime_error
		{
		public:
			Exception(std::string&& msg)
				: std::runtime_error(msg)
			{}
		};

		// @param neighborLength 近傍粒子半径
		// @param l_0 初期粒子間距離
		// @param minX 計算空間の最小座標
		// @param maxX 計算空間の最大座標
		Grid(const double neighborLength, const double l_0,
			const Vector& minX, const Vector& maxX)
			: blockLength(neighborLength), origin(minX),
			data(boost::extents
				[Ceil(maxX[0] - minX[0], neighborLength)+2] // 水平方向の最大ブロック数（近傍粒子探索の分を含める）
				[Ceil(maxX[1] - minX[1], neighborLength)+2] // 鉛直方向の最大ブロック数（近傍粒子探索の分を含める）
				[1 + (Ceil(neighborLength, l_0)+1)*(Ceil(neighborLength, l_0) + 1)]) // 1ブロック内の最大粒子数＋存在する粒子数
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

		// 対象の位置を含むブロックの水平方向の番号
		auto BlockX(const Vector& x) const
		{
			return Floor(x[0] - origin[0], blockLength);
		}

		// 対象の位置を含むブロックの鉛直方向の番号
		auto BlockZ(const Vector& x) const
		{
			return Floor(x[1] - origin[1], blockLength);
		}

		// 各ブロック内の粒子数
		auto ParticleCount(const Index i, const Index j) const
		{
			return data[i][j][0]; // 先頭は粒子数を格納してある
		}

		// 1ブロック内の最大粒子数
		auto MaxParticles() const
		{
			return static_cast<ParticleID>(data.shape()[2]) - 1; // 先頭の粒子数の格納分を除く
		}

		// 粒子を格納する
		bool Store(const Vector& x, const ParticleID particle)
		{
			const auto i = BlockX(x);
			const auto j = BlockZ(x);
			const auto n = GridSizeX();
			const auto m = GridSizeZ();

			// 領域外なら格納しない
			if(
				(0 <= i) && (i < n) &&
				(0 <= j) && (j < m))
			{
				const auto k = ParticleCount(i, j);
				const auto maxCount = MaxParticles();
				if(k >= maxCount)
				{
					throw Exception("Too many particle in a block");
				}
				Particle(i, j, static_cast<Index>(k)) = particle;
				ParticleCount(i, j) = k + 1;

				return true;
			}
			else
			{
				return false;
			}
		}

		// 近傍粒子イテレーター
		struct Iterator final : public std::iterator<std::input_iterator_tag, ParticleID>
		{
		private:
			using Block = std::tuple<Index, Index>;

			// 末尾の近傍ブロック番号（探索対象ブロックの総数）
			static constexpr std::size_t LAST_NEIGHBOR = MAX_NEIGHBOR_BLOCK;

			// 末尾も含めた近傍ブロックの総数
			static constexpr auto MAX_NEIGHBOR = LAST_NEIGHBOR + 1;

			// 参照先のグリッド
			const Grid& grid;

			// 探索対象ブロック
			const std::array<Block, LAST_NEIGHBOR> neighbor;

			// 今のイテレーターの遷移状態 = 粒子番号*LAST_NEIGHBOR_ID + 近傍ブロック番号
			std::size_t index;

			// begin用
			// @param g 参照先のグリッド
			// @param x 探索対象の位置
			// @param idx 遷移状態
			Iterator(const Grid& g, const Vector& x, const decltype(index) idx)
				: grid(g), neighbor{{
				std::make_tuple(g.BlockX(x) - 1, g.BlockZ(x) - 1),
				std::make_tuple(g.BlockX(x) - 1, g.BlockZ(x) + 0),
				std::make_tuple(g.BlockX(x) - 1, g.BlockZ(x) + 1),
				std::make_tuple(g.BlockX(x) + 0, g.BlockZ(x) - 1),
				std::make_tuple(g.BlockX(x) + 0, g.BlockZ(x) + 0),
				std::make_tuple(g.BlockX(x) + 0, g.BlockZ(x) + 1),
				std::make_tuple(g.BlockX(x) + 1, g.BlockZ(x) - 1),
				std::make_tuple(g.BlockX(x) + 1, g.BlockZ(x) + 0),
				std::make_tuple(g.BlockX(x) + 1, g.BlockZ(x) + 1), }},
				index(idx)
			{
				// 先頭ブロックが空なら次のブロックに移動しておく
				const auto neighborIndex = index%MAX_NEIGHBOR;
				const auto neighborBlock = neighbor[neighborIndex];
				const auto i = std::get<0>(neighborBlock);
				const auto j = std::get<1>(neighborBlock);
				auto count = grid.ParticleCount(i, j);
				if(count == 0)
				{
					Increment();
				}
			}

			// end用
			// @param g 参照先のグリッド
			// @param idx 遷移状態
			Iterator(const Grid& g, const decltype(index) idx)
				: grid(g), neighbor{}, index(idx)
			{}

			void Increment()
			{
				const auto neighborIndex = index%MAX_NEIGHBOR;
				const auto neighborBlock = neighbor[neighborIndex];
				const auto i = std::get<0>(neighborBlock);
				const auto j = std::get<1>(neighborBlock);
				const auto k = index / MAX_NEIGHBOR;

				const auto count = grid.ParticleCount(i, j);
				if(k + 1 >= count)
				{
					if(neighborIndex == LAST_NEIGHBOR - 1)
					{
						// 次が最終ブロックなら最終ブロックに移動するだけ
						index = LAST_NEIGHBOR;
					}
					else
					{
						// 最終ブロック以外なら、次の有効なブロックまで移動
						const auto n = grid.GridSizeX();
						const auto m = grid.GridSizeZ();
						bool isValid = false;
						for(index = neighborIndex; !isValid && (index + 1 < LAST_NEIGHBOR); index++)
						{
							const auto nextNeighborBlock = neighbor[index + 1];
							const auto nextI = std::get<0>(nextNeighborBlock);
							const auto nextJ = std::get<1>(nextNeighborBlock);

							// 範囲内かつ粒子数が存在するところのみ有効
							isValid =
								(0 <= nextI) && (nextI < n) &&
								(0 <= nextJ) && (nextJ < m);
							if(isValid)
							{
								const auto nextCount = grid.ParticleCount(nextI, nextJ);
								isValid = (nextCount > 0);
							}
						}
						if(!isValid) // 最後まで空のブロックが続くなら末尾に移動
						{
							index = LAST_NEIGHBOR;
						}
					}
				}
				else
				{
					index += MAX_NEIGHBOR; // 次の粒子に移動
				}
			}

		public:

			// 先頭イテレーターを作成
			// @param g 参照先のグリッド
			// @param x 探索対象の位置
			static auto CreateBegin(const Grid& g, const Vector& x)
			{
				return Iterator(g, x, decltype(index)(0) +
					((g.BlockX(x) == 0) ? 3 : 0) +
					((g.BlockZ(x) == 0) ? 1 : 0));
			}

			// 末尾イテレーターを作成
			// @param g 参照先のグリッド
			// @param x 探索対象の位置
			static auto CreateEnd(const Grid& g)
			{
				return Iterator(g, LAST_NEIGHBOR);
			}

			Iterator& operator=(const Iterator&) = delete;

			ParticleID operator*() const
			{
				const auto neighborIndex = index%MAX_NEIGHBOR;
				const auto neighborBlock = neighbor[neighborIndex];
				const auto i = std::get<0>(neighborBlock);
				const auto j = std::get<1>(neighborBlock);
				const auto k = static_cast<Index>(index / MAX_NEIGHBOR);

				return grid.Particle(i, j, k);
			}

			Iterator& operator++()
			{
				Increment();
				return *this;
			}

			bool operator==(const Iterator& it)
			{
				// 遷移状態が同じなら同じイテレーターとする
				return this->index == it.index;
			}
		};

		// 先頭イテレーターを作成
		auto cbegin(const Vector& x) const
		{
			return Iterator::CreateBegin(*this, x);
		}

		// 末尾イテレーターを作成
		auto cend() const
		{
			return Iterator::CreateEnd(*this);
		}
	};
}
#endif
