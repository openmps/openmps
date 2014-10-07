#ifndef GRID_INCLUDED
#define GRID_INCLUDED

#include <vector>
#include <map>
#include <boost/iterator/iterator_facade.hpp>
#include "Vector.hpp"

namespace OpenMps
{
	// 近傍粒子探索用グリッド
	class Grid
	{
	public:
		// ブロック（グリッドの1つ）の識別子
		typedef std::pair<int, int> BlockID;

	private:


		typedef std::vector<int> Block;


		std::map<BlockID, Block> grid;


		// ブロックの大きさ
		double blockSize;

	public:

		// 近傍粒子番号イテレータ
		struct NeighborBlockIterator
			: public boost::iterator_facade<NeighborBlockIterator, const BlockID, boost::forward_traversal_tag>
		{
			static const int INVALID_INDEX = -1;
			static const int MAX_INDEX = 10;

			// 探索対象（中心）ブロック
			BlockID id;

			// 遷移状態
			int index;

			friend class boost::iterator_core_access;

			// 次の近傍ブロック
			inline void increment()
			{
				index++;
			}

			// 一つ前の近傍ブロック
			inline void decrement()
			{
				index--;
			}

			// ブロックの参照
			inline const BlockID dereference() const
			{
				BlockID ret(id.first - 1 + (index % 3), id.second - 1 + (index / 3));
				return ret;
			}

			// 等号
			inline bool equal(const NeighborBlockIterator& other) const
			{
				return (this->index == other.index);
			}

			// コンストラクタは非公開
			NeighborBlockIterator(const std::map<BlockID, Block>& grid, const BlockID& block, int index)
				: id(block), index(index)
			{
			}

		public:

			// 最初の近傍ブロックを作成する
			// @param grid グリッド
			// @param block 自分のブロック
			inline static NeighborBlockIterator CreateBegin(const std::map<BlockID, Block>& grid, const BlockID& block)
			{
				NeighborBlockIterator it(grid, block, INVALID_INDEX);
				it.increment();

				return it;

			}

			// 最後の近傍ブロックを作成する
			// @param grid グリッド
			// @param block 自分のブロック
			inline static NeighborBlockIterator CreateEnd(const std::map<BlockID, Block>& grid, const BlockID& block)
			{
				NeighborBlockIterator it(grid, block, MAX_INDEX);
				it.decrement();

				return it;
			}
		};

		// @param blockSize 1ブロックの大きさ
		Grid(const double blockSize);

		// その位置が属するブロックの番号を取得する
		// @param 位置ベクトル
		// @return ブロック番号
		inline BlockID GetBlockID(const Vector& x) const
		{
			// ブロック番号を計算
			const int gridI = (int)std::ceil(x[0]/blockSize);
			const int gridJ = (int)std::ceil(x[1]/blockSize);
			return BlockID(gridI, gridJ);
		}

		// 粒子番号をグリッドに登録する
		// @param 登録する粒子
		// @param 粒子番号
		inline void AddParticle(const Vector& x, const int particleID)
		{
			// ブロック番号を計算
			auto id = GetBlockID(x);

			// 粒子番号を登録
			grid[id].push_back(particleID);
		}

		// 指定されたブロック番号のブロックを取得する
		// @param id ブロック番号
		inline Block operator[](const BlockID& id) const
		{
			// そのブロックがあれば
			auto it = this->grid.find(id);
			if(it != this->grid.end())
			{
				// ブロックを返す
				return it->second;
			}

			// なかったら空のブロック
			return Block();
		}


		inline NeighborBlockIterator begin(const Vector& x) const
		{
			return cbegin(x);
		}

		inline NeighborBlockIterator end(const Vector& x) const
		{
			return cend(x);
		}

		// 指定した位置にあるブロックの近傍ブロックのうち最初のものを取得する
		// @param ブロックの位置
		inline NeighborBlockIterator cbegin(const Vector& x) const
		{
			BlockID myBlock = this->GetBlockID(x);

			return NeighborBlockIterator::CreateBegin(this->grid, myBlock);
		}

		// 指定した位置にあるブロックの近傍ブロックのうち最後のものを取得する
		// @param ブロックの位置
		inline NeighborBlockIterator cend(const Vector& x) const
		{
			BlockID myBlock = this->GetBlockID(x);

			return NeighborBlockIterator::CreateEnd(this->grid, myBlock);
		}

		// グリッドを全消去する
		inline void Clear()
		{
			grid.clear();
		}
	};
}
#endif