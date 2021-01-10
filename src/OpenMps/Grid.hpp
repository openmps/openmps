﻿#ifndef GRID_INCLUDED
#define GRID_INCLUDED

#pragma warning(push, 0)
#include <iterator>
#include <array>
#include <memory>
#pragma warning(pop)

#include "Vector.hpp"

// OpenMP関連
#ifdef _OPENMP
	#if _OPENMP < 200805 // v3.0未満の場合、atomic captureが使えないので
		#define ATOMIC_LOCK
	#endif
#endif
#ifdef ATOMIC_LOCK
#include <omp.h>
#endif

namespace { namespace OpenMps
{
	// 近傍粒子探索用グリッド
	class Grid final
	{
	public:
		using ParticleID = std::size_t;

		using Index = std::size_t;

		// 探索対象ブロックの総数
#ifdef DIM3
		static constexpr std::size_t MAX_NEIGHBOR_BLOCK = 3 * 3 * 3; // 3次元なので
#else
		static constexpr std::size_t MAX_NEIGHBOR_BLOCK = 3 * 3; // 2次元なので
#endif

		// 最大ブロック数
		const std::array<const Index, DIM + 1> Size;

	private:
		static constexpr auto AXIS_PARTICLE = AXIS_Z + 1; // 粒子番号が入っている軸

		// 1ブロックの長さ（影響半径に等しい）
		const double blockLength;

		// 原点（計算空間の最大座標に等しい）
		const Vector origin;

		// 各ブロックの粒子番号
		std::unique_ptr<ParticleID[]> data;

#ifdef ATOMIC_LOCK
		// 登録粒子数計算用のスレッドロック
		std::unique_ptr<::omp_lock_t[]> locks;
#endif

		static auto Ceil(const double a, const double b)
		{
			return static_cast<Index>(std::ceil(a / b));
		}
		static auto Floor(const double a, const double b)
		{
			return static_cast<Index>(std::floor(a / b));
		}

		// 各ブロック内の粒子
		auto& Particle(const Index i,
#ifdef DIM3
			const Index j,
#endif
			const Index k, const Index l)
		{
			return data[((i
#ifdef DIM3
				* Size[AXIS_Y] + j
#endif
				) * Size[AXIS_Z] + k) * Size[AXIS_PARTICLE] + l + 1]; // 先頭は粒子数を格納してある
		}
		auto Particle(const Index i,
#ifdef DIM3
			const Index j,
#endif
			const Index k, const Index l) const
		{
			return data[((i
#ifdef DIM3
				* Size[AXIS_Y] + j
#endif
				) * Size[AXIS_Z] + k) * Size[AXIS_PARTICLE] + l + 1]; // 先頭は粒子数を格納してある
		}

		// 各ブロック内の粒子数
		auto& ParticleCount(const Index i,
#ifdef DIM3
			const Index j,
#endif
			const Index k)
		{
			return data[((i
#ifdef DIM3
				* Size[AXIS_Y] + j
#endif
				) * Size[AXIS_Z] + k) * Size[AXIS_PARTICLE] + 0]; // 先頭は粒子数を格納してある
		}

#ifdef ATOMIC_LOCK
		// 各ブロックの登録粒子数計算用のスレッドロック
		auto& Locks(const Index i,
#ifdef DIM3
			const Index j,
#endif
			const Index k)
		{
			return locks[(i
#ifdef DIM3
				* Size[AXIS_Y] + j
#endif
				) * Size[AXIS_Z] + k]; // 先頭は粒子数を格納してある
		}
#endif

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
			: Size({
				Ceil(maxX[AXIS_X] - minX[AXIS_X], neighborLength) + 2, // 水平方向の最大ブロック数（近傍粒子探索の分を含める）
#ifdef DIM3
				Ceil(maxX[AXIS_Y] - minX[AXIS_Y], neighborLength) + 2, // 奥行き方向の最大ブロック数（近傍粒子探索の分を含める）
#endif
				Ceil(maxX[AXIS_Z] - minX[AXIS_Z], neighborLength) + 2, // 鉛直方向の最大ブロック数（近傍粒子探索の分を含める）
#ifdef DIM3
				1 + (Ceil(neighborLength, l_0) + 1) * (Ceil(neighborLength, l_0) + 1) * (Ceil(neighborLength, l_0) + 1) }) // 1ブロック内の最大粒子数＋存在する粒子数
#else
				1 + (Ceil(neighborLength, l_0) + 1) * (Ceil(neighborLength, l_0) + 1) }), // 1ブロック内の最大粒子数＋存在する粒子数
#endif
			blockLength(neighborLength), origin(minX),
			data(std::make_unique<decltype(data)::element_type[]>(Size[AXIS_X]*Size[AXIS_Z]*Size[AXIS_Z+1]
#ifdef DIM3
				* Size[AXIS_Y]
#endif
			))
#ifdef ATOMIC_LOCK
			, locks(std::make_unique<decltype(locks)::element_type[]>(
				Size[AXIS_X] * Size[AXIS_Z]
#ifdef DIM3
				* Size[AXIS_Y]
#endif
			))
#endif
		{
#ifdef ATOMIC_LOCK
			const auto nx = Size[AXIS_X];
#ifdef DIM3
			const auto ny = Size[AXIS_Y];
#endif
			const auto nz = Size[AXIS_Z];
			for (auto i = decltype(nx){0}; i < nx; i++)
			{
#ifdef DIM3
				for (auto j = decltype(ny){0}; j < nx; j++)
				{
#endif
					for (auto k = decltype(nz){0}; k < nz; k++)
					{
#ifdef DIM3
						::omp_init_lock(&Locks(i, j, k));
#else
						::omp_init_lock(&Locks(i, k));
#endif
					}
#ifdef DIM3
				}
#endif
			}
#endif
		}

#ifdef ATOMIC_LOCK
		~Grid()
		{
			const auto nx = Size[AXIS_X];
#ifdef DIM3
			const auto ny = Size[AXIS_Y];
#endif
			const auto nz = Size[AXIS_Z];
			for (auto i = decltype(nx){0}; i < nx; i++)
			{
#ifdef DIM3
				for (auto j = decltype(ny){0}; j < nx; j++)
				{
#endif
					for (auto k = decltype(nz){0}; k < nz; k++)
					{
#ifdef DIM3
						::omp_destroy_lock(&Locks(i, j, k));
#else
						::omp_destroy_lock(&Locks(i, k));
#endif
					}
#ifdef DIM3
				}
#endif
			}
		}
#endif

		Grid(Grid&&) = default;
		Grid(const Grid&) = delete;
		Grid& operator = (const Grid&) = delete;

		// 全消去（全ブロックの粒子数を0にする）
		void Clear()
		{
			const auto nx = Size[AXIS_X];
#ifdef DIM3
			const auto ny = Size[AXIS_Y];
#endif
			const auto nz = Size[AXIS_Z];
			for (auto i = decltype(nx){0}; i < nx; i++)
			{
#ifdef DIM3
				for (auto j = decltype(ny){0}; j < nx; j++)
				{
#endif
					for (auto k = decltype(nz){0}; k < nz; k++)
					{
#ifdef DIM3
						ParticleCount(i, j, k) = 0;
#else
						ParticleCount(i, k) = 0;
#endif
					}
#ifdef DIM3
				}
#endif
			}
		}

		// 対象の位置を含むブロック番号
		template<decltype(AXIS_X) AXIS>
		auto Block(const Vector& x) const
		{
			return Floor(x[AXIS] - origin[AXIS], blockLength);
		}

		// 各ブロック内の粒子数
		auto ParticleCount(const Index i,
#ifdef DIM3
			const Index j,
#endif
			const Index k) const
		{
			return data[((i
#ifdef DIM3
				* Size[AXIS_Y] + j
#endif
				) * Size[AXIS_Z] + k) * Size[AXIS_PARTICLE] + 0]; // 先頭は粒子数を格納してある
		}

		// 1ブロック内の最大粒子数
		auto MaxParticles() const
		{
			return Size[AXIS_PARTICLE] - 1; // 先頭の粒子数の格納分を除く
		}

		// 粒子を格納する
		bool Store(const Vector& x, const ParticleID particle)
		{
			const auto i = Block<AXIS_X>(x); const auto nx = Size[AXIS_X];
#ifdef DIM3
			const auto j = Block<AXIS_Y>(x); const auto ny = Size[AXIS_Y];
#endif
			const auto k = Block<AXIS_Z>(x); const auto nz = Size[AXIS_Z];

			// 領域外なら格納しない
			if(
				(0 <= i) && (i < nx) &&
#ifdef DIM3
				(0 <= j) && (j < ny) &&
#endif
				(0 <= k) && (k < nz))
			{
				decltype(data)::element_type l;
#ifdef ATOMIC_LOCK
				{
					auto lock = &Locks(i,
#ifdef DIM3
						j,
#endif
						k);
					omp_set_lock(lock);
#else
#ifdef _OPENMP
					#pragma omp atomic capture
#endif
#endif
					l = ParticleCount(i,
#ifdef DIM3
						j,
#endif
						k)++;
#ifdef ATOMIC_LOCK
					omp_unset_lock(lock);
				}
#endif

				const auto maxCount = MaxParticles();
				if(l >= maxCount)
				{
					throw Exception("Too many particle in a block");
				}
#ifdef DIM3
				Particle(i, j, k, static_cast<Index>(l)) = particle;
#else
				Particle(i, k, static_cast<Index>(l)) = particle;
#endif

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
#ifdef DIM3
			using Block = std::tuple<Index, Index, Index>;
#else
			using Block = std::tuple<Index, Index>;
#endif

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
#ifdef DIM3
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) - 1, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) + 0, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Y>(x) + 1, g.Block<AXIS_Z>(x) + 1),
#else
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) - 1, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 0, g.Block<AXIS_Z>(x) + 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Z>(x) - 1),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Z>(x) + 0),
				std::make_tuple(g.Block<AXIS_X>(x) + 1, g.Block<AXIS_Z>(x) + 1),
#endif
				}},
				index(idx)
			{
				// 先頭ブロックが空なら次のブロックに移動しておく
				const auto neighborIndex = index%MAX_NEIGHBOR;
				const auto neighborBlock = neighbor[neighborIndex];
				const auto i = std::get<AXIS_X>(neighborBlock);
#ifdef DIM3
				const auto j = std::get<AXIS_Y>(neighborBlock);
#endif
				const auto k = std::get<AXIS_Z>(neighborBlock);

#ifdef DIM3
				auto count = grid.ParticleCount(i, j, k);
#else
				auto count = grid.ParticleCount(i, k);
#endif
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
				const auto i = std::get<AXIS_X>(neighborBlock);
#ifdef DIM3
				const auto j = std::get<AXIS_Y>(neighborBlock);
#endif
				const auto k = std::get<AXIS_Z>(neighborBlock);
				const auto l = index / MAX_NEIGHBOR;

#ifdef DIM3
				const auto count = grid.ParticleCount(i, j, k);
#else
				const auto count = grid.ParticleCount(i, k);
#endif
				if(l + 1 >= count)
				{
					if(neighborIndex == LAST_NEIGHBOR - 1)
					{
						// 次が最終ブロックなら最終ブロックに移動するだけ
						index = LAST_NEIGHBOR;
					}
					else
					{
						// 最終ブロック以外なら、次の有効なブロックまで移動
						const auto nx = grid.Size[AXIS_X];
#ifdef DIM3
						const auto ny = grid.Size[AXIS_Y];
#endif
						const auto nz = grid.Size[AXIS_Z];
						bool isValid = false;
						for(index = neighborIndex; !isValid && (index + 1 < LAST_NEIGHBOR); index++)
						{
							const auto nextNeighborBlock = neighbor[index + 1];
							const auto nextI = std::get<AXIS_X>(nextNeighborBlock);
#ifdef DIM3
							const auto nextJ = std::get<AXIS_Y>(nextNeighborBlock);
#endif
							const auto nextK = std::get<AXIS_Z>(nextNeighborBlock);

							// 範囲内かつ粒子数が存在するところのみ有効
							isValid =
								(0 <= nextI) && (nextI < nx) &&
#ifdef DIM3
								(0 <= nextJ) && (nextJ < ny) &&
#endif
								(0 <= nextK) && (nextK < nz);
							if(isValid)
							{
#ifdef DIM3
								const auto nextCount = grid.ParticleCount(nextI, nextJ, nextK);
#else
								const auto nextCount = grid.ParticleCount(nextI, nextK);
#endif
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
#ifdef DIM3
					3 * 3 * ((g.Block<AXIS_X>(x) == 0) ? 1 : 0) +
					    3 * ((g.Block<AXIS_Y>(x) == 0) ? 1 : 0) +
#else
					    3 * ((g.Block<AXIS_X>(x) == 0) ? 1 : 0) +
#endif
					    1 * ((g.Block<AXIS_Z>(x) == 0) ? 1 : 0));
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
				const auto i = std::get<AXIS_X>(neighborBlock);
#ifdef DIM3
				const auto j = std::get<AXIS_Y>(neighborBlock);
#endif
				const auto k = std::get<AXIS_Z>(neighborBlock);
				const auto l = static_cast<Index>(index / MAX_NEIGHBOR);

#ifdef DIM3
				return grid.Particle(i, j, k, l);
#else
				return grid.Particle(i, k, l);
#endif
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
}}
#endif
