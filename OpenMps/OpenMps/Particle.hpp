#ifndef PARTICLE_INCLUDED
#define PARTICLE_INCLUDED
#include "defines.hpp"

#include "Vector.hpp"
#include "Grid.hpp"
#include <numeric>

namespace OpenMps
{
	// 粒子
	class Particle
	{
	public:
		// 粒子リスト
		typedef std::vector<Particle> List;

		// 粒子タイプ
		typedef enum
		{
			// 非圧縮性ニュートン流体（水など）
			ParticleTypeIncompressibleNewton,

			// 壁面（位置と速度が変化しない）
			ParticleTypeWall,

			// ダミー（粒子数密度の計算にのみ対象となる）
			ParticleTypeDummy,
		} ParticleType;

	private:
		// 粒子タイプの数
		static const int ParticleTypeMaxCount = 3;

#ifndef PRESSURE_EXPLICIT
		// 対象の粒子の圧力方程式の生成項の寄与分を計算する関数ポインタの型
		typedef double(Particle::*GetPpeSourceToFunc)(const Particle& particle_i, const double r_e, const double dt, const double n0) const;

		// 通常粒子を対象とした圧力方程式の生成項の寄与分を計算する
		double GetPpeSourceToNormal(const Particle& particle_i, const double r_e, const double dt, const double n0) const
		{
			namespace ublas = boost::numeric::ublas;

			// MPS-HS：b_i = Σ-1/dt/n0 r_e/r^3 u・x
			const auto u_ij = this->u - particle_i.u;
			const auto x_ij = this->x - particle_i.x;
			const double r_ij = ublas::norm_2(x_ij);
			const double ux = ublas::inner_prod(u_ij, x_ij);

			// 自分からの寄与分は0
			return ((r_ij == 0) ? 0 : -1.0/dt/n0 * r_e/(r_ij*r_ij*r_ij)*ux);
		}

		// 自分に対する圧力方程式の寄与分が0である粒子を対象とした、対象の粒子の圧力方程式の生成項の寄与分を計算する
		double GetPpeSourceToZero(const Particle&, const double, const double, const double) const
		{
			return 0;
		}

		// 各粒子タイプで対象の粒子の圧力方程式の生成項の寄与分を計算する関数
		static constexpr GetPpeSourceToFunc GetPpeSourceToFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetPpeSourceToNormal, // 水
			&Particle::GetPpeSourceToNormal, // 壁
			&Particle::GetPpeSourceToZero,   // ダミー
		};


		// 自分を対象とした圧力方程式の係数を計算する関数ポインタの型
		typedef double(Particle::*GetPpeMatrixTargetFunc)(const Particle& source, const double n0, const double r_e,
#ifndef MPS_HL
			const double lambda,
#endif
			const double rho) const;

		// 通常粒子を対象とした圧力方程式の係数を計算する
		double GetPpeMatrixTargetNormal(const Particle& source, const double n0, const double r_e,
#ifndef MPS_HL
			const double lambda,
#endif
			const double rho) const
		{
#ifndef MPS_HL
			// 標準MPS法：-2D/ρλ w/n0
			double w = this->Weight(source, r_e);
			return -2*DIM/(rho*lambda) * w/n0;
#else
			namespace ublas = boost::numeric::ublas;

			// MPS-HL：-1/ρ (5-D)/n0 r_e/r^3
			const auto x_ij = this->x - source.x;
			const double r_ij = ublas::norm_2(x_ij);

			return -1.0/rho * (5-DIM)/n0 *r_e/(r_ij*r_ij*r_ij);
#endif
		}

		// 自分に対する圧力方程式の係数が0である粒子を対象とした、圧力方程式の係数を計算する
		double GetPpeMatrixTargetZero(const Particle&, const double, const double,
#ifndef MPS_HL
			const double,
#endif
			const double) const
		{
			return 0;
		}
#endif
		// 各粒子タイプで自分を対象とした圧力方程式の係数を計算する関数
		static constexpr GetPpeMatrixTargetFunc GetPpeMatrixTargetFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetPpeMatrixTargetNormal, // 水
			&Particle::GetPpeMatrixTargetNormal, // 壁
			&Particle::GetPpeMatrixTargetZero,   // ダミー
		};


		// 対象の粒子へ与える粘性項を計算する関数ポインタの型
		typedef Vector(Particle::*ViscosityToFunc)(const Particle& particle_i, const double n_0, const double r_e,
#ifndef MPS_HV
			const double lambda,
#endif 
			const double nu) const;

		// 通常粒子へ与える粘性項を計算する
		Vector ViscosityToNormal(const Particle& particle_i, const double n_0, const double r_e,
#ifndef MPS_HV
			const double lambda,
#endif
			const double nu) const
		{
#ifndef MPS_HV
			// 標準MPS法：ν*2D/λn0 (u_j - u_i) w（ただし自分自身からは影響を受けない）
			Vector result = (nu * 2*DIM/lambda/n_0 * particle_i.Weight(*this, r_e))*(this->u - particle_i.u);
#else
			namespace ublas = boost::numeric::ublas;

			// MPS-HV：ν*(5-D)/n0 r_e/r^3 * (u_j - u_i)（ただし自分自身からは影響を受けない）
			const auto x_ij = this->x - particle_i.x;
			const double r_ij = ublas::norm_2(x_ij);
			Vector result = ((r_ij == 0) ? VectorZero
				: (nu * (5-DIM)/n_0 *r_e/(r_ij*r_ij*r_ij)*(this->u - particle_i.u)));
#endif
			return result;
		}

		// 対象の粒子へ粘性効果を与えない粒子の与える粘性項を計算する
		Vector ViscosityToZero(const Particle&, const double, const double,
#ifndef MPS_HV
			const double,
#endif
			const double) const
		{
			return VectorZero;
		}

		// 各粒子タイプで自分を対象とした圧力方程式の係数を計算する関数
		static constexpr ViscosityToFunc ViscosityToFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::ViscosityToNormal, // 水
			&Particle::ViscosityToNormal, // 壁
			&Particle::ViscosityToZero,   // ダミー
		};

		// 対象の粒子へ与える圧力勾配を計算する関数ポインタの型
		typedef Vector(Particle::*PressureGradientToFunc)(
			const Particle& particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP,
#endif
			const double r_e, const double dt, const double rho, const double n0) const;

		// 通常粒子へ与える圧力勾配を計算する
		Vector PressureGradientToNormal(
			const Particle& particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP,
#endif
			const double r_e, const double dt, const double rho, const double n0) const
		{
			namespace ublas = boost::numeric::ublas;

			auto dx = this->x - particle_i.x;
			auto r2 = ublas::inner_prod(dx, dx);

			// 標準MPS法：-Δt/ρ D/n_0 (p_j + p_i)/r^2 w * dx（ただし自分自身からは影響を受けない）
#ifdef PRESSURE_GRADIENT_MIDPOINT
			auto result = -(dt/rho * DIM/n0 * (this->p + particle_i.p)/r2 * particle_i.Weight(*this, r_e));
#else
			auto result = -(dt/rho * DIM/n0 * (this->p - minP        )/r2 * particle_i.Weight(*this, r_e));
#endif
			return (r2 == 0 ? 0 : result) * dx;
		}

		// 対象の粒子へ圧力勾配を与えない粒子の圧力勾配を計算する
		Vector PressureGradientToZero(
			const Particle&,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double,
#endif
			const double, const double, const double , const double) const
		{
			return VectorZero;
		}

		// 各粒子タイプで自分を対象とした圧力方程式の係数を計算する関数
		static constexpr PressureGradientToFunc PressureGradientToFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::PressureGradientToNormal, // 水
			&Particle::PressureGradientToNormal, // 壁
			&Particle::PressureGradientToZero,   // ダミー
		};


		// 粒子を加速（速度を変更）する関数ポインタの型
		typedef void(Particle::*AccelerateFunc)(const Vector& du);

		// 通常粒子を加速（速度を変更）する
		void AccelerateNormal(const Vector& du)
		{
			u += du;
		}

		// 移動しない粒子を加速（速度を変更）する
		void AccelerateZero(const Vector&)
		{
			// 動かさない
		}

		// 各粒子タイプで粒子を加速（速度を変更）する関数
		static constexpr AccelerateFunc AccelerateFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::AccelerateNormal, // 水
			&Particle::AccelerateZero,   // 壁
			&Particle::AccelerateZero,   // ダミー
		};


		// 粒子を移動（位置を変更）する関数ポインタの型
		typedef void(Particle::*MoveFunc)(const Vector& dx);

		// 通常粒子を移動（位置を変更）する
		void MoveNormal(const Vector& dx)
		{
			x += dx;
		}

		// 移動しない粒子を移動（位置を変更）する
		void MoveZero(const Vector&)
		{
			// 動かさない
		}

		// 各粒子タイプで粒子を移動（位置を変更）する関数
		static constexpr MoveFunc MoveFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::MoveNormal, // 水
			&Particle::MoveZero,   // 壁
			&Particle::MoveZero,   // ダミー
		};


		// 粘性項を計算する関数ポインタの型
		typedef Vector(Particle::*GetViscosityFunc)(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e,
#ifndef MPS_HV
			const double lambda,
#endif
			const double nu) const;

		// 通常粒子の粘性項を計算するする
		Vector GetViscosityNormal(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e,
#ifndef MPS_HV
			const double lambda,
#endif
			const double nu) const
		{
			// 粘性項を計算
			auto vis = std::accumulate(grid.cbegin(this->x), grid.cend(this->x), VectorZero,
				[this, &n_0, &r_e,
#ifndef MPS_HV
				&lambda,
#endif
				&nu, &particles, &grid](const Vector& sum, const Grid::BlockID block)
			{
				// 近傍ブロック内の粒子を取得
				auto neighbors = grid[block];

				// 近傍ブロック内の粒子に対して計算
				Vector duBlock = std::accumulate(neighbors.cbegin(), neighbors.cend(), VectorZero,
					[this, &n_0, &r_e,
#ifndef MPS_HV
					&lambda,
#endif
					&nu, &particles](const Vector& sum2, const int& idd)
				{
					const unsigned int id = static_cast<unsigned int>(idd);

					Vector duParticle = particles[id].ViscosityTo(*this, n_0, r_e,
#ifndef MPS_HV
						lambda,
#endif
						nu);
					return static_cast<Vector>(sum2 + duParticle);
				});

				return static_cast<Vector>(sum + duBlock);
			});

			// 粘性項を返す
			return vis;
		}

		// 移動しない粒子の粘性項を計算する
		Vector GetViscosityZero(const Particle::List&, const Grid&, const double, const double,
#ifndef MPS_HV
			const double,
#endif
			const double) const
		{
			return VectorZero;
		}

		// 各粒子タイプで粘性項を計算する関数
		static constexpr GetViscosityFunc GetViscosityFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetViscosityNormal, // 水
			&Particle::GetViscosityZero,   // 壁
			&Particle::GetViscosityZero,   // ダミー
		};


		// 重み関数を計算する関数ポインタの型
		typedef double(Particle::*WeightFunc)(const Particle& target, const double r_e) const;

		// 通常粒子の重み関数を計算する
		double WeightNormal(const Particle& target, const double r_e) const
		{
			return target.WeightTarget(*this, r_e);
		}

		// 重み関数を計算しない粒子の重み関数を計算する
		double WeightZero(const Particle&, const double) const
		{
			return 0;
		}

		// 各粒子タイプで重み関数を計算する関数
		static constexpr WeightFunc WeightFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::WeightNormal, // 水
			&Particle::WeightNormal, // 壁
			&Particle::WeightZero,   // ダミー
		};


		// 粘性項を計算する関数ポインタの型
		typedef void(Particle::*UpdateNeighborDensityFunc)(const Particle::List& particles, const Grid& grid, const double r_e);

		// 通常粒子の粘性項を計算する
		void UpdateNeighborDensityNormal(const Particle::List& particles, const Grid& grid, const double r_e)
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


		// 移動しない粒子の粘性項を計算する
		void UpdateNeighborDensityZero(const Particle::List&, const Grid&, const double)
		{
			// 計算しない
			n = 0;
		}

		// 各粒子タイプで粘性項を計算する関数
		static constexpr UpdateNeighborDensityFunc UpdateNeighborDensityFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::UpdateNeighborDensityNormal, // 水
			&Particle::UpdateNeighborDensityNormal, // 壁
			&Particle::UpdateNeighborDensityZero,   // ダミー
		};


#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する関数ポインタの型
		typedef Vector(Particle::*GetCorrectionByTooNearFunc)(const Particle::List& particles, const Grid& grid, const double r_e, const double rho, const double tooNearRatio, const double tooNearCoefficient) const;

		// 通常粒子の過剰接近粒子からの速度補正量を計算する
		Vector GetCorrectionByTooNearNormal(const Particle::List& particles, const Grid& grid, const double r_e, const double rho, const double tooNearLength, const double tooNearCoefficient) const
		{
			// 運動量を計算
			auto p_i = rho * this->u;

			Vector du = std::accumulate(grid.cbegin(this->x), grid.cend(this->x), VectorZero,
				[this, &r_e, &rho, &tooNearLength, &tooNearCoefficient, &p_i, &particles, &grid](const Vector& sum, const Grid::BlockID block)
			{
				// 近傍ブロック内の粒子を取得
				auto neighbors = grid[block];

				Vector duBlock = std::accumulate(neighbors.cbegin(), neighbors.cend(), VectorZero,
					[this, &r_e, &rho, &tooNearLength, &tooNearCoefficient, &p_i, &particles](const Vector& sum2, const int& idd)
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
					if ((0 < r_ij) & (r_ij < tooNearLength))
					{
						// 合成運動量を計算
						auto p = p_i + rho * particles[id].u;

						// 運動量の変化量を計算
						auto delta_p = p_i - p / 2;
						auto abs_delta_p = ublas::inner_prod(delta_p, x_ij) / r_ij;

						// 運動量が増加する方向なら
						if (abs_delta_p > 0)
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

		// 移動しない粒子の過剰接近粒子からの速度補正量を計算する
		Vector GetCorrectionByTooNearZero(const Particle::List&, const Grid&, const double, const double, const double, const double) const
		{
			return VectorZero;
		}
		// 各粒子タイプで過剰接近粒子からの速度補正量を計算する関数
		static constexpr GetCorrectionByTooNearFunc GetCorrectionByTooNearFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetCorrectionByTooNearNormal, // 水
			&Particle::GetCorrectionByTooNearZero,   // 壁
			&Particle::GetCorrectionByTooNearZero,   // ダミー
		};
#endif


#ifdef PRESSURE_EXPLICIT
		// 圧力を計算する関数ポインタの型
		typedef void(Particle::*UpdatePressureFunc)(const double c, const double rho0, const double n0);

		// 通常粒子の圧力を計算する
		void UpdatePressureNormal(const double c, const double rho0, const double n0)
		{
			// 仮想的な密度：ρ0/n0 * n
			auto rho = rho0/n0 * n;

			// 圧力の計算：c^2 (ρ-ρ0)（基準密度以下なら圧力は発生しない）
			p = (rho <= rho0) ? 0 : c*c*(rho - rho0);
		}

		// 圧力を持たない粒子の圧力を計算する
		void UpdatePressureZero(const double, const double, const double)
		{
			// 計算しない
			p = 0;
		}

		// 各粒子タイプで圧力を計算する関数
		static const UpdatePressureFunc UpdatePressureFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::UpdatePressureNormal, // 水
			&Particle::UpdatePressureNormal, // 壁
			&Particle::UpdatePressureZero,   // ダミー
		};
#else


		// 圧力方程式の生成項を計算する関数ポインタの型
		typedef double(Particle::*GetPpeSourceFunc)(
#ifdef MPS_HS
			const Particle::List& particles, const Grid& grid, const double r_e,
#endif
			const double n0, const double dt, const double surfaceRatio) const;

		// 通常粒子の圧力方程式の生成項を計算する
		double GetPpeSourceNormal(
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
				: (n - n0) / n0 / (dt*dt);
#endif
		}

		// 圧力を持たない粒子の圧力方程式の生成項を計算する
		double GetPpeSourceZero(
#ifdef MPS_HS
			const Particle::List&, const Grid&, const double,
#endif
			const double, const double, const double) const
		{
			// 計算しない
			return 0;
		}

		// 各粒子タイプで圧力方程式の生成項を計算する関数
		static constexpr GetPpeSourceFunc GetPpeSourceFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetPpeSourceNormal, // 水
			&Particle::GetPpeSourceNormal, // 壁
			&Particle::GetPpeSourceZero,   // ダミー
		};


		// 圧力方程式の係数を計算する関数ポインタの型
		typedef double(Particle::*GetPpeMatrixFunc)(const Particle& target, const double n0, const double r_e,
#ifndef MPS_HL
			const double lambda,
#endif
			const double rho) const;

		// 通常粒子の圧力方程式の係数を計算する
		double GetPpeMatrixNormal(const Particle& target, const double n0, const double r_e,
#ifndef MPS_HL
			const double lambda,
#endif
			const double rho) const
		{
			return target.GetPpeMatrixTarget(*this, n0, r_e,
#ifndef MPS_HL
					lambda,
#endif
					rho);
		}

		// 圧力を持たない粒子の圧力方程式の係数を計算する
		double GetPpeMatrixZero(const Particle&, const double, const double,
#ifndef MPS_HL
			const double,
#endif
			const double) const
		{
			// 計算しない
			return 0;
		}
#endif

		// 各粒子タイプで圧力方程式の係数を計算する関数
		static constexpr GetPpeMatrixFunc GetPpeMatrixFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetPpeMatrixNormal, // 水
			&Particle::GetPpeMatrixNormal, // 壁
			&Particle::GetPpeMatrixZero,   // ダミー
		};

		// 圧力勾配を計算する関数ポインタの型
		typedef Vector(Particle::*GetPressureGradientFunc)(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0) const;

		// 通常粒子の圧力勾配を計算する
		Vector GetPressureGradientNormal(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0)  const
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

		// 移動しない粒子の圧力勾配を計算する
		Vector GetPressureGradientZero(const Particle::List&, const Grid&, const double, const double, const double, const double) const
		{
			// 計算しない
			return VectorZero;
		}

		// 各粒子タイプで圧力勾配を計算する関数
		static constexpr GetPressureGradientFunc GetPressureGradientFuncTable[ParticleTypeMaxCount] =
		{
			&Particle::GetPressureGradientNormal, // 水
			&Particle::GetPressureGradientZero,   // 壁
			&Particle::GetPressureGradientZero,   // ダミー
		};

	protected:
		// 位置ベクトル
		Vector x;

		// 速度ベクトル
		Vector u;

		// 圧力
		double p;

		// 粒子数密度
		double n;

		// 粒子タイプ
		const ParticleType type;

		// 自分を対象とした重み関数を計算する
		// @param source 基準とする粒子
		// @param r_e 影響半径
		double WeightTarget(const Particle& source, const double r_e) const
		{
			namespace ublas = boost::numeric::ublas;

			// 2粒子間の距離から重み関数の値を返す
			auto r = ublas::norm_2(source.x - this->x);
			return Particle::Weight(r, r_e);
		}

#ifndef PRESSURE_EXPLICIT
		// 対象の粒子の圧力方程式の生成項の寄与分を計算する
		// @param source 基準とする粒子
		// @param dt 時間刻み
		// @param r_e 影響半径
		// @@aram n0 基準粒子数密度
		double GetPpeSourceTo(const Particle& particle_i, const double r_e, const double dt, const double n0) const
		{
			return (this->*(Particle::GetPpeSourceToFuncTable[type]))(particle_i, r_e, dt, n0);
		}

		// 自分を対象とした圧力方程式の係数を計算する
		// @param source 基準とする粒子
#ifndef MPS_HL
		// @param lambda 拡散モデル係数λ
#endif
		// @param rho 密度
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		double GetPpeMatrixTarget(const Particle& source, const double n0, const double r_e,
#ifndef MPS_HL
			const double lambda,
#endif
			const double rho) const
		{
			return (this->*(Particle::GetPpeMatrixTargetFuncTable[type]))(source, n0, r_e,
#ifndef MPS_HL
				lambda,
#endif
				rho);
		}
#endif

		// 対象の粒子へ与える粘性項を計算する
		// @param particle_i 対象の粒子粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
#ifndef MPS_HV
		// @param lambda 拡散モデル係数λ
#endif
		// @param nu 粘性係数
		Vector ViscosityTo(const Particle& particle_i, const double n_0, const double r_e,
#ifndef MPS_HV
			const double lambda,
#endif
			const double nu) const
		{
			return (this->*(Particle::ViscosityToFuncTable[type]))(particle_i, n_0, r_e,
#ifndef MPS_HV
				lambda,
#endif
				nu);
		}

		// 対象の粒子へ与える圧力勾配を計算する
		// @param particle_i 対象の粒子
#ifndef PRESSURE_GRADIENT_MIDPOINT
		// @param minP 計算で使用する自分の圧力（周囲の最小圧力）
#endif
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		Vector PressureGradientTo(
			const Particle& particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP,
#endif
			const double r_e, const double dt, const double rho, const double n0) const
		{
			return (this->*(Particle::PressureGradientToFuncTable[type]))(particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
				minP,
#endif
				r_e, dt, rho, n0);
		}

		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param u 速度ベクトルの水平方向成分
		// @param w 速度ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		Particle(const ParticleType type, const double x, const double z, const double u, const double w, const double p, const double n)
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

	public:
		virtual ~Particle()
		{
		}

		// 距離から重み関数を計算する
		// @param r 距離
		// @param r_e 影響半径
		static double Weight(const double r, const double r_e)
		{
			// 影響半径内ならr_e/r-1を返す（ただし距離0の場合は0）
			return ((0 < r) && (r < r_e)) ? (r_e/r - 1) : 0;
		}

		// 粒子を加速（速度を変更）する
		// @param du 速度の変化量
		void Accelerate(const Vector& du)
		{
			(this->*(Particle::AccelerateFuncTable[type]))(du);
		}

		// 粒子を移動（位置を変更）する
		// @param dx 位置の変化量
		void Move(const Vector& dx)
		{
			(this->*(Particle::MoveFuncTable[type]))(dx);
		}

		// 粘性項を計算する
		// @param particles 粒子リスト
		// @param grid 近傍粒子探索用グリッド
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
#ifndef MPS_HV
		// @param lambda 拡散モデル係数λ
#endif
		// @param nu 粘性係数
		Vector GetViscosity(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e,
#ifndef MPS_HV
			const double lambda,
#endif
			const double nu) const
		{
			return (this->*(Particle::GetViscosityFuncTable[type]))(particles, grid, n_0, r_e,
#ifndef MPS_HV
				lambda,
#endif
				nu);
		}

		// 重み関数を計算する
		// @param target 計算相手の粒子
		// @param r_e 影響半径
		double Weight(const Particle& target, const double r_e) const
		{
			return (this->*(Particle::WeightFuncTable[type]))(target, r_e);
		}

		// 粒子数密度を計算する
		// @param particles 粒子リスト
		// @param grid 近傍粒子探索用グリッド
		// @param r_e 影響半径
		void UpdateNeighborDensity(const Particle::List& particles, const Grid& grid, const double r_e)
		{
			return (this->*(Particle::UpdateNeighborDensityFuncTable[type]))(particles, grid, r_e);
		}

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する
		// @param particles 粒子リスト
		// @param grid 近傍粒子探索用グリッド
		// @param r_e 影響半径
		// @param rho 密度
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		Vector GetCorrectionByTooNear(const Particle::List& particles,const Grid& grid, const double r_e, const double rho, const double tooNearRatio, const double tooNearCoefficient) const
		{
			return (this->*(Particle::GetCorrectionByTooNearFuncTable[type]))(particles, grid, r_e, rho, tooNearRatio, tooNearCoefficient);
		}
#endif

#ifdef PRESSURE_EXPLICIT
		// 圧力を計算する
		// @param c 音速
		// @param rho0 （基準）密度
		// @param n0 基準粒子数密度
		void UpdatePressure(const double c, const double rho0, const double n0)
		{
			return (this->*(Particle::UpdatePressureFuncTable[type]))(c, rho0, n0);
		}
#else
		// 圧力方程式の生成項を計算する
#ifdef MPS_HS
		// @param particles 粒子リスト
		// @param grid 近傍粒子探索用グリッド
		// @param r_e 影響半径
#endif
		// @param n0 基準粒子数密度
		// @param dt 時間刻み
		// @param surfaceRatio 自由表面の判定係数（基準粒子数密度からのずれがこの割合以下なら自由表面と判定される）
		double GetPpeSource(
#ifdef MPS_HS
			const Particle::List& particles, const Grid& grid, const double r_e,
#endif
			const double n0, const double dt, const double surfaceRatio) const
		{
			return (this->*(Particle::GetPpeSourceFuncTable[type]))(
#ifdef MPS_HS
				particles, grid, r_e,
#endif
				n0, dt, surfaceRatio);
		}

		// 圧力方程式の係数を計算する
		// @param particle 対象粒子
		// @param n_0 基準粒子数密度
		// @param r_e 影響半径
#ifndef MPS_HL
		// @param lambda 拡散モデル係数λ
#endif
		// @param rho 密度
		double GetPpeMatrix(const Particle& target, const double n0, const double r_e,
#ifndef MPS_HL
			const double lambda,
#endif
			const double rho) const
		{
			return (this->*(Particle::GetPpeMatrixFuncTable[type]))(target, n0, r_e,
#ifndef MPS_HL
				lambda,
#endif
				rho);
		}
#endif

		// 圧力勾配を計算する
		// @param particles 粒子リスト
		// @param grid 近傍粒子探索用グリッド
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 基準粒子数密度
		Vector GetPressureGradient(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0) const
		{
			return (this->*(Particle::GetPressureGradientFuncTable[type]))(particles, grid, r_e, dt, rho, n0);
		}

		////////////////
		// プロパティ //
		////////////////

		// 位置ベクトルの水平方向成分を取得する
		inline double X() const
		{
			return x[0];
		}

		// 位置ベクトルの鉛直成分を取得する
		inline double Z() const
		{
			return x[1];
		}

		// 速度ベクトルの水平方向成分を取得する
		inline double U() const
		{
			return u[0];
		}

		// 速度ベクトルの鉛直方向成分を取得する
		inline double W() const
		{
			return u[1];
		}

		// 圧力を取得する
		inline double P() const
		{
			return p;
		}

		// 圧力を設定する
		// @param value 設定値
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		inline void P(const double value, const double n0, const double surfaceRatio)
		{
			// 負圧であったり自由表面の場合は圧力0
			p = ((value < 0) || IsSurface(n0, surfaceRatio)) ? 0 : value;
		}

		// 粒子数密度を取得する
		inline double N() const
		{
			return n;
		}

		// 種類を取得する
		inline ParticleType Type() const
		{
			return type;
		}

		// 位置ベクトルを取得する
		inline Vector VectorX() const
		{
			return x;
		}

		// 速度ベクトルを取得する
		inline Vector VectorU() const
		{
			return u;
		}

		// 自由表面かどうかの判定
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		inline bool IsSurface(const double n0, const double surfaceRatio) const
		{
			return n/n0 < surfaceRatio;
		}

		// 代入演算子
		// @param src 代入元
		Particle& operator=(const Particle& src)
		{
			this->x = src.x;
			this->u = src.u;
			this->p = src.p;
			this->n = src.n;
			const_cast<ParticleType&>(this->type) = src.type;
		}
	};

	// 非圧縮性ニュートン流体（水など）
	class ParticleIncompressibleNewton : public Particle
	{
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param u 速度ベクトルの水平方向成分
		// @param w 速度ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		ParticleIncompressibleNewton(const double x, const double z, const double u, const double w, const double p, const double n)
			: Particle(ParticleTypeIncompressibleNewton, x, z, u, w, p, n)
		{
		}
	};

	// 壁粒子（位置と速度が変化しない）
	class ParticleWall : public Particle
	{
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		ParticleWall(const double x, const double z, const double p, const double n)
			: Particle(ParticleTypeWall, x, z, 0, 0, p, n)
		{
		}
	};

	// ダミー粒子（粒子数密度の計算にのみ対象となる）
	class ParticleDummy : public Particle
	{
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		ParticleDummy(const double x, const double z)
			: Particle(ParticleTypeDummy, x, z, 0, 0, 0, 0)
		{
		}
	};
}
#endif
