#include "Particle.hpp"
#include <numeric>
#include <boost/numeric/ublas/io.hpp>

namespace OpenMps
{
	Particle::Particle(const double x, const double z, const double u, const double w, const double p, const double n)
	{
		// 各物理値を初期化
		this->x[0] = x;
		this->x[1] = z;
		this->u[0] = u;
		this->u[1] = w;
		this->p = p;
		this->n = n;
	}

	void Particle::UpdateNeighborDensity(const Particle::List& particles, const double r_e)
	{
		// 重み関数の総和を粒子数密度とする
		// TODO: 全粒子探索してるのでGet
		n = std::accumulate(particles.cbegin(), particles.cend(), 0.0,
			[this, &r_e](const double sum, const Particle::Ptr& particle)
			{
				double w = this->Weight(*particle, r_e);
				return sum + w;
			});
	}

	Vector Particle::GetViscosity(const Particle::List& particles, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const
	{
		// 粘性項を計算して返す
		return std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
			[this, &n_0, &r_e, &lambda, &nu, &dt](const Vector& sum, const Particle::Ptr& particle)
			{
				Vector du = particle->ViscosityTo(*this, n_0, r_e, lambda, nu, dt);
				return (Vector)(sum + du);
			});
	}

	Vector Particle::GetPressureGradient(const Particle::List& particles, const double r_e, const double dt, const double rho, const double n0)  const
	{
		Vector du;

#ifdef PRESSURE_GRADIENT_MIDPOINT
		// 速度修正量を計算
		du = std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
			[this, &r_e, &dt, &rho, &n0](const Vector& sum, const Particle::Ptr& particle)
			{
				auto du = particle->PressureGradientTo(*this, r_e, dt, rho, n0);
				return (Vector)(sum + du);
			});
#else
		// 最小圧力を取得する
		auto minPparticle = std::min_element(particles.cbegin(), particles.cend(),
			[](const Particle::Ptr& base, const Particle::Ptr& target)
			{
				return base->p < target->p;
			});

		// 速度修正量を計算
		du = std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
			[this, &r_e, &dt, &rho, &n0, &minPparticle](const Vector& sum, const Particle::Ptr& particle)
			{
				auto du = particle->PressureGradientTo(*this, (*minPparticle)->p, r_e, dt, rho, n0);
				return (Vector)(sum + du);
			});
#endif
		return du;
	}

#ifdef MODIFY_TOO_NEAR
	Vector Particle::GetCorrectionByTooNear(const Particle::List& particles, const double r_e, const double rho, const double tooNearLength, const double tooNearCoefficient) const
	{
		// 運動量を計算
		auto p_i = rho * this->u;

		Vector du = std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
			[this, &r_e, &rho, &tooNearLength, & tooNearCoefficient, &p_i, &zero](const Vector& sum, const Particle::Ptr& particle)
			{
				namespace ublas = boost::numeric::ublas;

				// 相対距離を計算
				auto x_ij = particle->x - this->x;
				double r_ij = ublas::norm_2(x_ij);

				// 相対距離が過剰接近なら
				auto d = zero;
				if((0 < r_ij) & (r_ij < tooNearLength))
				{
					// 合成運動量を計算
					auto p = p_i + rho * particle->u;

					// 運動量の変化量を計算
					auto delta_p = p_i - p/2;
					auto abs_delta_p = ublas::inner_prod(delta_p, x_ij) / r_ij;

					// 運動量が増加する方向なら
					if(abs_delta_p > 0)
					{
						// 反発率をかける
						auto p_m = (tooNearCoefficient * abs_delta_p / r_ij) * x_ij;

						// 速度の修正量を計算
						d = p_m / rho;
					}
				}

				return (Vector)(sum - d);
			});

		return du;
	}
#endif
}