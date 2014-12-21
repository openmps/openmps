#include "Particle.hpp"
#pragma warning(push, 0)
#include <numeric>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/numeric/ublas/io.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#pragma warning(pop)

namespace OpenMps
{
#ifndef PRESSURE_EXPLICIT
	const Particle::GetPpeSourceToFunc Particle::GetPpeSourceToFuncTable[] =
	{
		&Particle::GetPpeSourceToNormal, // 水
		&Particle::GetPpeSourceToNormal, // 壁
		&Particle::GetPpeSourceToZero,   // ダミー
	};

	const Particle::GetPpeMatrixTargetFunc Particle::GetPpeMatrixTargetFuncTable[] =
	{
		&Particle::GetPpeMatrixTargetNormal, // 水
		&Particle::GetPpeMatrixTargetNormal, // 壁
		&Particle::GetPpeMatrixTargetZero,   // ダミー
	};
#endif

	const Particle::ViscosityToFunc Particle::ViscosityToFuncTable[] =
	{
		&Particle::ViscosityToNormal, // 水
		&Particle::ViscosityToNormal, // 壁
		&Particle::ViscosityToZero,   // ダミー
	};

	const Particle::PressureGradientToFunc Particle::PressureGradientToFuncTable[] =
	{
		&Particle::PressureGradientToNormal, // 水
		&Particle::PressureGradientToNormal, // 壁
		&Particle::PressureGradientToZero,   // ダミー
	};

	const Particle::AccelerateFunc Particle::AccelerateFuncTable[] =
	{
		&Particle::AccelerateNormal, // 水
		&Particle::AccelerateZero,   // 壁
		&Particle::AccelerateZero,   // ダミー
	};

	const Particle::MoveFunc Particle::MoveFuncTable[] =
	{
		&Particle::MoveNormal, // 水
		&Particle::MoveZero,   // 壁
		&Particle::MoveZero,   // ダミー
	};

	const Particle::GetViscosityFunc Particle::GetViscosityFuncTable[] =
	{
		&Particle::GetViscosityNormal, // 水
		&Particle::GetViscosityZero,   // 壁
		&Particle::GetViscosityZero,   // ダミー
	};

	const Particle::WeightFunc Particle::WeightFuncTable[] =
	{
		&Particle::WeightNormal, // 水
		&Particle::WeightNormal, // 壁
		&Particle::WeightZero,   // ダミー
	};

	const Particle::UpdateNeighborDensityFunc Particle::UpdateNeighborDensityFuncTable[] =
	{
		&Particle::UpdateNeighborDensityNormal, // 水
		&Particle::UpdateNeighborDensityNormal, // 壁
		&Particle::UpdateNeighborDensityZero,   // ダミー
	};

#ifdef MODIFY_TOO_NEAR
	const Particle::GetCorrectionByTooNearFunc Particle::GetCorrectionByTooNearFuncTable[] =
	{
		&Particle::GetCorrectionByTooNearNormal, // 水
		&Particle::GetCorrectionByTooNearZero,   // 壁
		&Particle::GetCorrectionByTooNearZero,   // ダミー
	};
#endif

#ifdef PRESSURE_EXPLICIT
	const Particle::UpdatePressureFunc Particle::UpdatePressureFuncTable[] =
	{
		&Particle::UpdatePressureNormal, // 水
		&Particle::UpdatePressureNormal, // 壁
		&Particle::UpdatePressureZero,   // ダミー
	};
#else
	const Particle::GetPpeSourceFunc Particle::GetPpeSourceFuncTable[] =
	{
		&Particle::GetPpeSourceNormal, // 水
		&Particle::GetPpeSourceNormal, // 壁
		&Particle::GetPpeSourceZero,   // ダミー
	};

	const Particle::GetPpeMatrixFunc Particle::GetPpeMatrixFuncTable[] =
	{
		&Particle::GetPpeMatrixNormal, // 水
		&Particle::GetPpeMatrixNormal, // 壁
		&Particle::GetPpeMatrixZero,   // ダミー
	};
#endif

	const Particle::GetPressureGradientFunc Particle::GetPressureGradientFuncTable[] =
	{
		&Particle::GetPressureGradientNormal, // 水
		&Particle::GetPressureGradientZero,   // 壁
		&Particle::GetPressureGradientZero,   // ダミー
	};

	Particle::Particle(const ParticleType type, const double x, const double z, const double u, const double w, const double p, const double n)
		:type(type)
	{
		// 各物理値を初期化
		this->x[0] = x;
		this->x[1] = z;
		this->u[0] = u;
		this->u[1] = w;
		this->p = p;
		this->n = n;
	}

	void Particle::UpdateNeighborDensityNormal(const Particle::List& particles, const Grid& grid, const double r_e)
	{
		// 重み関数の総和を粒子数密度とする
		n = std::accumulate(grid.cbegin(this->x), grid.cend(this->x), 0.0,
			[this, &r_e, &particles, &grid](const double sum, const Grid::BlockID block)
			{
				// 近傍ブロック内の粒子を取得
				auto neighbors = grid[block];

				// 近傍ブロック内の粒子に対して計算
				return sum + std::accumulate(neighbors.cbegin(), neighbors.cend(), 0.0,
					[this, &r_e, &particles](const double sum2, const int& idd)
					{
						const unsigned int id = static_cast<unsigned int>(idd);

						double w = this->Weight(particles[id], r_e);
						return sum2 + w;
					});
			});
	}

	Vector Particle::GetViscosityNormal(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e, const double lambda, const double nu) const
	{
		// 粘性項を計算
		auto vis = std::accumulate(grid.cbegin(this->x), grid.cend(this->x), VectorZero,
			[this, &n_0, &r_e, &lambda, &nu, &particles, &grid](const Vector& sum, const Grid::BlockID block)
			{
				// 近傍ブロック内の粒子を取得
				auto neighbors = grid[block];

				// 近傍ブロック内の粒子に対して計算
				Vector duBlock = std::accumulate(neighbors.cbegin(), neighbors.cend(), VectorZero,
					[this, &n_0, &r_e, &lambda, &nu, &particles](const Vector& sum2, const int& idd)
					{
						const unsigned int id = static_cast<unsigned int>(idd);

						Vector duParticle = particles[id].ViscosityTo(*this, n_0, r_e, lambda, nu);
						return static_cast<Vector>(sum2 + duParticle);
					});

				return static_cast<Vector>(sum + duBlock);
			});

		// 粘性項を返す
		return vis;
	}

	Vector Particle::GetPressureGradientNormal(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0)  const
	{
#ifdef PRESSURE_GRADIENT_MIDPOINT
		// 速度修正量を計算
		Vector du = std::accumulate(grid.cbegin(this->x), grid.cend(this->x), VectorZero,
			[this, &r_e, &dt, &rho, &n0, &particles, &grid](const Vector& sum, const Grid::BlockID block)
			{
				// 近傍ブロック内の粒子を取得
				auto neighbors = grid[block];

				// 近傍ブロック内の粒子に対して計算
				Vector duBlock = std::accumulate(neighbors.cbegin(), neighbors.cend(), VectorZero,
					[this, &r_e, &dt, &rho, &n0, &particles](const Vector& sum2, const int& idd)
					{
						const unsigned int id = static_cast<unsigned int>(idd);

						auto duParticle = particles[id].PressureGradientTo(*this, r_e, dt, rho, n0);
						return (Vector)(sum2 + duParticle);
					});

				return static_cast<Vector>(sum + duBlock);
			});
#else
		// 最小圧力を取得する
		auto minPparticle = std::min_element(particles.cbegin(), particles.cend(),
			[](const Particle& base, const Particle& target)
			{
				return base.p < target.p;
			});
		const double minP = (*minPparticle).p;

		// 速度修正量を計算
		Vector du = std::accumulate(particles.cbegin(), particles.cend(), zero,
			[this, &r_e, &dt, &rho, &n0, &minP](const Vector& sum, const Particle& particle)
			{
				auto du = particle.PressureGradientTo(*this, minP, r_e, dt, rho, n0);
				return (Vector)(sum + du);
			});
#endif
		return du;
	}

#ifdef MODIFY_TOO_NEAR
	Vector Particle::GetCorrectionByTooNearNormal(const Particle::List& particles, const Grid& grid, const double r_e, const double rho, const double tooNearLength, const double tooNearCoefficient) const
	{
		// 運動量を計算
		auto p_i = rho * this->u;

		Vector du = std::accumulate(grid.cbegin(this->x), grid.cend(this->x), VectorZero,
			[this, &r_e, &rho, &tooNearLength, & tooNearCoefficient, &p_i, &particles, &grid](const Vector& sum, const Grid::BlockID block)
			{
				// 近傍ブロック内の粒子を取得
				auto neighbors = grid[block];

				Vector duBlock = std::accumulate(neighbors.cbegin(), neighbors.cend(), VectorZero,
					[this, &r_e, &rho, &tooNearLength, & tooNearCoefficient, &p_i, &particles](const Vector& sum2, const int& idd)
					{
						const unsigned int id = static_cast<unsigned int>(idd);

						namespace ublas = boost::numeric::ublas;

						// 相対距離を計算
						auto x_ij = particles[id].x - this->x;
						double r_ij = ublas::norm_2(x_ij);

						// 相対距離が過剰接近なら
						Vector d;
						d(0) = 0;
						d(1) = 0;
						if((0 < r_ij) & (r_ij < tooNearLength))
						{
							// 合成運動量を計算
							auto p = p_i + rho * particles[id].u;

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

						return static_cast<Vector>(sum2 - d);
					});

				return static_cast<Vector>(sum + duBlock);
			});

		return du;
	}
#endif

	double Particle::GetPpeSourceNormal(
#ifdef MPS_HS
		const Particle::List& particles, const Grid& grid, const double r_e,
#endif
		const double n0, const double dt, const double surfaceRatio) const
	{
		// 自由表面の場合は0
		return IsSurface(n0, surfaceRatio) ? 0
#ifdef MPS_HS
			// MPS-HS：b_i = Σ-1/dt/n0 r_e/r^3 u・x
			: std::accumulate(grid.cbegin(this->x), grid.cend(this->x), 0.0,
				[this, &r_e, &dt, &n0, &particles, &grid](const double& sum, const Grid::BlockID block)
				{
					// 近傍ブロック内の粒子を取得
					auto neighbors = grid[block];

					// 近傍ブロック内の粒子に対して計算
					const double value = std::accumulate(neighbors.cbegin(), neighbors.cend(), 0.0,
						[this, &r_e, &dt, &n0, &particles](const double& sum2, const int& idd)
						{
							const unsigned int id = static_cast<unsigned int>(idd);

							const double value2 = particles[id].GetPpeSourceTo(*this, r_e, dt, n0);

							return sum2 + value2;
						});

					return sum + value;
				});
#else
			// 標準MPS法：b_i = 1/dt^2 * (n_i - n0)/n0
			: (n - n0)/n0 /(dt*dt);
#endif
	}
}
