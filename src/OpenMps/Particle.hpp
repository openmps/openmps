#ifndef PARTICLE_INCLUDED
#define PARTICLE_INCLUDED

#pragma warning(push, 0)
#include <numeric>
#pragma warning(pop)

#include "defines.hpp"
#include "Vector.hpp"

namespace { namespace OpenMps
{
	// 粒子
	class Particle final
	{
	public:
		// 粒子の種類
		enum class Type
		{
			// 非圧縮性ニュートン流体（水など）
			IncompressibleNewton,

			// 壁面
			Wall,

			// ダミー粒子
			Dummy,

			// 無効粒子
			Disabled,
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
		Type type;

	public:
		Particle(const Type t)
			: x(VectorZero),
			u(VectorZero),
			p(0),
			n(0),
			type(t)
		{}

		Particle(const Particle& src) = default;
		Particle(Particle&& src) = default;
		Particle& operator=(const Particle& src) = default;
		Particle& operator=(Particle&& src) = default;

		// 粒子を無効化する
		void Disable()
		{
			this->type = Type::Disabled;
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
}}
#endif
