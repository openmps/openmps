#ifndef PARTICLE_INCLUDED
#define PARTICLE_INCLUDED

#include "defines.hpp"
#include "Vector.hpp"
#include <numeric>

namespace OpenMps
{
	// 粒子
	class Particle
	{
	public:
		// 粒子の種類
		enum class Type
		{
			IncompressibleNewton,
			Wall,
			Dummy,
		};

	private:
		// 位置ベクトル
		Vector x;

		// 速度ベクトル
		Vector u;

		// 圧力
		double p;

		// 粒子数密度
		double n;

		// 粒子の種類
		const Type type;

	public:
		Particle(const Type t)
			: x(VectorZero),
			u(VectorZero),
			p(0),
			n(0),
			type(t)
		{}

		Particle(const Particle& src)
			: x(src.x),
			u(src.u),
			p(src.p),
			n(src.n),
			type(src.type)
		{}

		Particle(Particle&& src)
			: x(std::move(src.x)),
			u(std::move(src.u)),
			p(src.p),
			n(src.n),
			type(src.type)
		{}

		Particle& operator=(const Particle& src)
		{
			this->x = src.x;
			this->u = src.u;
			this->p = src.p;
			this->n = src.n;
			const_cast<Type&>(this->type) = src.type;
		}

		Particle& operator=(Particle&& src)
		{
			this->x = std::move(src.x);
			this->u = std::move(src.u);
			this->p = src.p;
			this->n = src.n;
			const_cast<Type&>(this->type) = src.type;
		}

		// 距離から重み関数を計算する
		// @param r 距離
		// @param r_e 影響半径
		static double W(const double r, const double r_e)
		{
			// 影響半径内ならr_e/r-1を返す（ただし距離0の場合は0）
			return ((0 < r) && (r < r_e)) ? (r_e / r - 1) : 0;
		}

		////////////////
		// プロパティ //
		////////////////

		// 位置ベクトル
		const auto& X() const
		{
			return x;
		}
		auto& X()
		{
			return x;
		}

		// 速度ベクトル
		const auto& U() const
		{
			return u;
		}
		auto& U()
		{
			return u;
		}

		// 圧力
		const auto& P() const
		{
			return p;
		}
		auto& P()
		{
			return p;
		}

		// 粒子数密度
		const auto& N() const
		{
			return n;
		}
		auto& N()
		{
			return n;
		}

		// 種類
		const auto& TYPE() const
		{
			return type;
		}
	};
}
#endif
