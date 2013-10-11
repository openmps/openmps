#include "Particle.hpp"
#include <numeric>

namespace OpenMps
{
	Particle::Particle(const double& x, const double& z, const double& u, const double& w, const double& p, const double& n)
	{
		// 各物理値を初期化
		this->x[0] = x;
		this->x[1] = z;
		this->u[0] = u;
		this->u[1] = w;
		this->p = p;
		this->n = n;
	}

	void Particle::UpdateNeighborDensity(const Particle::List& particles, const double& r_e)
	{
		// 重み関数の総和を粒子数密度とする
		// TODO: 全粒子探索してるのでGet
		n = std::accumulate(particles.cbegin(), particles.cend(), 0.0,
			[this, &r_e](const double& sum, const Particle::Ptr& particle)
			{
				double w = this->Weight(*particle, r_e);
				return sum + w;
			});
	}

	Vector Particle::GetViscosity(const Particle::List& particles, const double& n_0, const double& r_e, const double& lambda, const double& nu, const double& dt) const
	{
		Vector zero;
		zero(0) = 0;
		zero(1) = 0;

		// 粘性項を計算して返す
		return std::accumulate(particles.cbegin(), particles.cend(), zero,
			[this, &n_0, &r_e, &lambda, &nu, &dt](const Vector& sum, const Particle::Ptr& particle)
			{
				Vector du = particle->ViscosityTo(*this, n_0, r_e, lambda, nu, dt);
				return (Vector)(sum + du);
			});
	}	

	Vector Particle::GetPressureGradient(const Particle::List& particles, const double& r_e, const double& dt, const double& rho, const double& n0)
	{
		Vector zero;
		zero(0) = 0;
		zero(1) = 0;

		// 最小圧力を取得する
		auto minPparticle = std::min_element(particles.cbegin(), particles.cend(),
			[](const Particle::Ptr& base, const Particle::Ptr& target)
			{
				return base->p < target->p;
			});

		// 速度修正量を計算して返す
		return std::accumulate(particles.cbegin(), particles.cend(), zero,
			[this, &r_e, &dt, &rho, &n0, &minPparticle](const Vector& sum, const Particle::Ptr& particle)
			{
				auto du = particle->PressureGradientTo(*this, (*minPparticle)->p, r_e, dt, rho, n0);
				return (Vector)(sum + du);
			});
	}	
}