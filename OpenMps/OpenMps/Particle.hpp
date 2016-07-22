#ifndef PARTICLE_INCLUDED
#define PARTICLE_INCLUDED

#include "defines.hpp"
#include "Interaction.hpp"

namespace OpenMps
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
		PS::F64vec x;

		// 速度ベクトル
		PS::F64vec u;

		// 圧力
		double p;

		// 粒子数密度
		double n;

		// 粒子の種類
		Type type;

		// 影響半径
		double r_e;

		// 作業バッファー
		Force force;

	public:
		Particle()
			: x(0, 0),
			u(0, 0),
			p(0),
			n(0),
			type(Type::Disabled),
			r_e(0)
		{}

		Particle(const Type t, const double re)
			: x(0, 0),
			u(0, 0),
			p(0),
			n(0),
			type(t),
			r_e(re)
		{}

		Particle(const Particle& src)
			: x(src.x),
			u(src.u),
			p(src.p),
			n(src.n),
			type(src.type),
			r_e(src.r_e)
		{}

		Particle(Particle&& src)
			: x(std::move(src.x)),
			u(std::move(src.u)),
			p(src.p),
			n(src.n),
			type(src.type),
			r_e(src.r_e)
		{}

		Particle& operator=(const Particle& src)
		{
			Copy(src);
			return *this;
		}

		Particle& operator=(Particle&& src)
		{
			this->x = std::move(src.x);
			this->u = std::move(src.u);
			this->p = src.p;
			this->n = src.n;
			this->type = src.type;
			this->r_e = src.r_e;

			return *this;
		}

		// 距離から重み関数を計算する
		// @param r 距離
		static double W(const double r, const double r_e)
		{
			// 影響半径内ならr_e/r-1を返す（ただし距離0の場合は0）
			return ((0 < r) && (r < r_e)) ? (r_e / r - 1) : 0;
		}

		void Copy(const Particle& src)
		{
			this->x = src.x;
			this->u = src.u;
			this->p = src.p;
			this->n = src.n;
			this->type = src.type;
			this->r_e = src.r_e;
		}
		void copyFromFP(const Particle& src)
		{
			Copy(src);
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
		auto getPos() const
		{
			return X();
		}
		void setPos(const decltype(x)& src)
		{
			x = src;
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
		void copyFromForce(const ParticleNumberDensity& src)
		{
			n = src.val;
		}

		// 種類
		const auto& TYPE() const
		{
			return type;
		}

		// 影響半径
		const auto& R_e() const
		{
			return r_e;
		}
		auto getRSearch() const
		{
			return R_e();
		}

		// 力
		const auto& A() const
		{
			return force.val;
		}
		const auto& Du() const
		{
			return force.val;
		}
		void copyFromForce(const Force& src)
		{
			force.val = src.val;
		}
	};
}
#endif
