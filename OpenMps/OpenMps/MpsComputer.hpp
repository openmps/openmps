#ifndef MPSCOMPUTER_INCLUDED
#define MPSCOMPUTER_INCLUDED
#pragma warning(push, 0)
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#pragma warning(pop)

#include "Particle.hpp"
#include "MpsEnvironment.hpp"

namespace OpenMps
{
	// MPS法による計算空間
	class MpsComputer
	{
	private:
		// 線形方程式用の疎行列
		typedef boost::numeric::ublas::compressed_matrix<double> Matrix;

		// 線形方程式用の多次元ベクトル
		typedef boost::numeric::ublas::vector<double> LongVector;

		// 粒子リスト
		std::vector<Particle> particles;

		// 計算空間のパラメーター
		MpsEnvironment environment;

#ifndef PRESSURE_EXPLICIT
		// 圧力方程式
		struct Ppe
		{
			// 係数行列
			Matrix A;

			// 圧力方程式の未知数
			LongVector x;

			// 圧力方程式の右辺
			LongVector b;

			// 収束判定（許容誤差）
			double allowableResidual;

			// 共役勾配法で使う用
			struct ConjugateGradient
			{
				// 残差ベクトル
				LongVector r;

				// 探索方向ベクトル
				LongVector p;

				// 係数行列と探索方向ベクトルの積
				LongVector Ap;
			} cg;
		} ppe;
#endif

		// 速度修正量
		std::vector<Vector> du;

		// 2粒子間の距離を計算する
		// @param p1 粒子1
		// @param p2 粒子2
		static double R(const Particle& p1, const Particle& p2)
		{
			const auto r = p1.X() - p2.X();
			return std::sqrt(boost::numeric::ublas::inner_prod(r, r));
		}

		// 自由表面かどうかの判定
		// @param n 粒子数密度
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		static bool IsSurface(const double n, const double n0, const double surfaceRatio)
		{
			return n / n0 < surfaceRatio;
		}

		// 時間刻みを決定する
		void DetermineDt()
		{
			namespace ublas = boost::numeric::ublas;

			// 最大速度を取得
			const auto maxUParticle = *std::max_element(particles.cbegin(), particles.cend(),
				[](const Particle& base, const Particle& target)
			{
				const auto& baseU = base.U();
				const auto& targetU = target.U();
				return (ublas::inner_prod(baseU, baseU) < ublas::inner_prod(targetU, targetU));
			});
			const auto maxU = ublas::norm_2(maxUParticle.U());

			// 時間刻みを設定
			environment.SetDt(maxU);
		}

		// 粒子数密度を計算する
		void ComputeNeighborDensities()
		{
			const double r_e = environment.R_e;

			// 全粒子で
			for (auto& particle : particles)
			{
				// ダミー粒子を除く
				if(particle.TYPE() != Particle::Type::Dummy)
				{
					// 粒子数密度を計算する
					particle.N() = std::accumulate(particles.cbegin(), particles.cend(), 0.0,
						[&particle, &r_e](const double sum, const Particle& neighbor)
					{
						const double w = Particle::W(R(particle, neighbor), r_e);
						return sum + w;
					});
				}
			}
		}

		// 陽的にで解く部分（第一段階）を計算する
		void ComputeExplicitForces()
		{
			const auto n0 = environment.N0();
			const auto r_e = environment.R_e;
			const auto lambda = environment.Lambda();
			const auto nu = environment.Nu;
			const auto dt = environment.Dt();
			const auto g = environment.G;

			// 加速度を全初期化
			auto& a = du;
			a.clear();

			// 全粒子で
			for (auto& particle : particles)
			{	
				// 重力の計算
				auto d = g;

				// 水粒子のみ
				if(particle.TYPE() == Particle::Type::IncompressibleNewton)
				{
					// 粘性の計算
					auto vis = std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
						[&particle, &n0, &r_e, &lambda, &nu](const Vector sum, const Particle& neighbor)
					{
						// 標準MPS法：ν*2D/λn0 (u_j - u_i) w（ただし自分自身からは影響を受けない）
						const double w = (neighbor.TYPE() == Particle::Type::Dummy) ? 0 : Particle::W(R(particle, neighbor), r_e);
						const auto du = (w == 0) ? VectorZero : ((nu * 2 * DIM / lambda / n0 * w)*(neighbor.U() - particle.U()));
						return static_cast<Vector>(sum + du);
					});

					d += vis;
				}
				a.push_back(d);
			}

			// 全粒子で
			for (unsigned int i = 0; i < particles.size(); i++)
			{
				// 水粒子のみ
				if(particles[i].TYPE() == Particle::Type::IncompressibleNewton)
				{
					// 位置・速度を修正
					particles[i].U() += a[i] * dt;
					particles[i].X() += particles[i].U()* dt;
				}
			}
		}

		// 陰的にで解く部分（第ニ段階）を計算する
		void ComputeImplicitForces()
		{
#ifdef PRESSURE_EXPLICIT
			// 得た圧力を計算する
			for(unsigned int i = 0; i < particles.size(); i++)
			{
				const auto c = environment.C;
				const auto n0 = environment.N0();
				const auto rho = environment.Rho;

				auto& particle = particles[i];

				particle->UpdatePressure(c, rho, n0);
			}
#else
			// 圧力方程式を設定
			SetPressurePoissonEquation();

			// 圧力方程式を解く
			SolvePressurePoissonEquation();

			// 得た圧力を代入する
			for (unsigned int i = 0; i < particles.size(); i++)
			{
				const auto n0 = environment.N0();
				const auto surfaceRatio = environment.SurfaceRatio;

				// 負圧であったり自由表面の場合は圧力0
				const auto p = ppe.x(i);
				const auto n = particles[i].N();
				particles[i].P() = ((p < 0) || IsSurface(n, n0, surfaceRatio)) ? 0 : p;
			}
#endif

			// 圧力勾配項を計算する
			ModifyByPressureGradient();
		}

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子を補正する
		void ModifyTooNear()
		{
			// 速度修正量を全初期化
			du.clear();

			// 全粒子で
			for (auto& particle : particles)
			{
				const auto r_e = environment.R_e;
				const auto rho = environment.Rho;
				const auto tooNearLength = environment.TooNearLength;
				const auto tooNearCoefficient = environment.TooNearCoefficient;

				// 過剰接近粒子からの速度修正量を計算する
				Vector d = particle->GetCorrectionByTooNear(particles, r_e, rho, tooNearLength, tooNearCoefficient);
				du.push_back(d);
			}

			// 全粒子で
			for (unsigned int i = 0; i < particles.size(); i++)
			{
				// 位置・速度を修正
				const auto dt = environment.Dt();
				Vector thisDu = du[i];
				particles[i]->Accelerate(thisDu);
				particles[i]->Move(thisDu * dt);
			}
		}
#endif

#ifndef PRESSURE_EXPLICIT
		// 圧力方程式を設定する
		void SetPressurePoissonEquation()
		{
			const auto n0 = environment.N0();
			const auto dt = environment.Dt();
			const auto surfaceRatio = environment.SurfaceRatio;
			const auto r_e = environment.R_e;
			const auto rho = environment.Rho;
			const auto lambda = environment.Lambda();

			// 粒子数を取得
			const std::size_t n = particles.size();

			// 粒子に増減があれば
			if (n != ppe.b.size())
			{
				// サイズを変えて作り直し
				ppe.A = Matrix(n, n);
				ppe.x = LongVector(n);
				ppe.b = LongVector(n);
				ppe.cg.r = LongVector(n);
				ppe.cg.p = LongVector(n);
				ppe.cg.Ap = LongVector(n);
			}
			// 全粒子で
			for (unsigned int i = 0; i < n; i++)
			{
				const auto thisN = particles[i].N();

				// ダミー粒子と自由表面は0
				if((particles[i].TYPE() == Particle::Type::Dummy) || IsSurface(thisN, n0, surfaceRatio))
				{
					ppe.b(i) = 0;
					ppe.x(i) = 0;
				}
				else
				{
					// 生成項を計算する
					// 標準MPS法：b_i = 1/dt^2 * (n_i - n0)/n0
					const auto b_i = (thisN - n0) / n0 / (dt*dt);
					ppe.b(i) = b_i;

					// 圧力を未知数ベクトルの初期値にする
					const auto x_i = particles[i].P();
					ppe.x(i) = x_i;
				}
			}
			// TODO: 以下もそうだけど、圧力方程式を作る際にインデックス指定のfor回さなきゃいけないのが気持ち悪いので、どうにかしたい

			// 係数行列初期化
			ppe.A.clear();

			// 全粒子で
			for (unsigned int i = 0; i < n; i++)
			{
				// ダミー粒子と自由表面は対角項だけ1
				if((particles[i].TYPE() == Particle::Type::Dummy) || IsSurface(particles[i].N(), n0, surfaceRatio))
				{
					ppe.A(i, i) = 1.0;
				}
				else
				{
					// 対角項を初期化
					double a_ii = 0;

					// 他の粒子に対して
					// TODO: 全粒子探索してるので遅い
					for(unsigned int j = 0; j < particles.size(); j++)
					{
						// ダミー粒子と自分以外
						if((particles[j].TYPE() != Particle::Type::Dummy) && (i != j))
						{
							// 非対角項を計算
							// 標準MPS法：-2D/ρλ w/n0
							const auto a_ij = -2 * DIM / (rho*lambda) * Particle::W(R(particles[i], particles[j]), r_e) / n0;

							// 自由表面の場合は非対角項は設定しない
							if(!IsSurface(particles[j].N(), n0, surfaceRatio))
							{
								ppe.A(i, j) = a_ij;
							}

							// 対角項も設定
							a_ii -= a_ij;
						}

					}

					// 対角項を設定
					ppe.A(i, i) = a_ii;
				}
			}
		}

		// 圧力方程式をを解く
		void SolvePressurePoissonEquation()
		{
			// 共役勾配法で解く
			// TODO: 前処理ぐらい入れようよ
			namespace ublas = boost::numeric::ublas;

			auto& A = ppe.A;
			auto& x = ppe.x;
			auto& b = ppe.b;
			auto& r = ppe.cg.r;
			auto& p = ppe.cg.p;
			auto& Ap = ppe.cg.Ap;

			// 初期値を設定
			//  (Ap)_0 = A * x
			//  r_0 = b - Ap
			//  p_0 = r_0
			//  rr = r・r
			Ap = ublas::prod(A, x);
			r = b - Ap;
			p = r;
			double rr = ublas::inner_prod(r, r);
			const double residual0 = rr*ppe.allowableResidual*ppe.allowableResidual;

			// 初期値で既に収束している場合は即時終了
			bool isConverged = (residual0 == 0);
			// 未知数分だけ繰り返す
			for (unsigned int i = 0; (i < x.size()) && (!isConverged); i++)
			{
				// 計算を実行
				//  Ap = A * p
				//  α = rr/(p・Ap)
				//  x' += αp
				//  r' -= αAp
				//  r'r' = r'・r'
				Ap = ublas::prod(A, p);
				const double alpha = rr / ublas::inner_prod(p, Ap);
				x += alpha * p;
				r -= alpha * Ap;
				const double rrNew = ublas::inner_prod(r, r);

				// 収束判定
				const double residual = rrNew;
				isConverged = (residual < residual0);

				// 収束していなければ、残りの計算を実行
				if (!isConverged)
				{
					// 残りの計算を実行
					//  β= r'r'/rr
					//  p = r' + βp
					//  rr = r'r'
					const double beta = rrNew / rr;
					p = r + beta * p;
					rr = rrNew;
				}
			}

			// 理論上は未知数分だけ繰り返せば収束するはずだが、収束しなかった場合は
			if (!isConverged)
			{
				// どうしようもないので例外
				Exception exception;
				exception.Message = "Conjugate Gradient method couldn't solve Pressure Poison Equation";
				throw exception;
			}
		};
#endif

		// 対象の粒子へ与える圧力勾配を計算する
		// @param particle_i 対象の粒子
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		static Vector PressureGradientTo(const Particle& particle, const Particle& neighbor, const double r_e, const double dt, const double rho, const double n0)
		{
			namespace ublas = boost::numeric::ublas;

			// 標準MPS法：-Δt/ρ D/n_0 (p_j + p_i)/r^2 w * dx（ただし自分自身からは影響を受けない）
			const auto dx = neighbor.X() - particle.X();
			const auto r2 = ublas::inner_prod(dx, dx);
			const auto result = -dt / rho * DIM / n0 * (neighbor.P() + particle.P()) / r2 * Particle::W(R(neighbor, particle), r_e);
			return r2 == 0 ? VectorZero : (result * dx);
		}

		// 圧力勾配によって速度と位置を修正する
		void ModifyByPressureGradient()
		{
			// 速度修正量を全初期化
			du.clear();

			// 全粒子で
			for (auto& particle : particles)
			{
				const double r_e = environment.R_e;
				const double dt = environment.Dt();
				const double rho = environment.Rho;
				const double n0 = environment.N0();

				// 圧力勾配を計算する
				Vector d;
				{
#ifdef PRESSURE_GRADIENT_MIDPOINT
					// 速度修正量を計算
					d = std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
						[&particle, &r_e, &dt, &rho, &n0](const Vector& sum, const Particle& neighbor)
					{
						auto du = PressureGradientTo(particle, neighbor, r_e, dt, rho, n0);
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
					d = std::accumulate(particles.cbegin(), particles.cend(), VectorZero,
						[this, &r_e, &dt, &rho, &n0, &minPparticle](const Vector& sum, const Particle::Ptr& particle)
					{
						auto du = particle->PressureGradientTo(*this, (*minPparticle)->p, r_e, dt, rho, n0);
						return (Vector)(sum + du);
					});
#endif
				}
				du.push_back(d);
			}

			// 全粒子で
			for (unsigned int i = 0; i < particles.size(); i++)
			{
				const double dt = environment.Dt();

				// 位置・速度を修正
				Vector thisDu = du[i];
				particles[i].U() += thisDu;
				particles[i].X() += thisDu * dt;
			}
		}

	public:
		struct Exception
		{
			std::string Message;
		};

#ifndef PRESSURE_EXPLICIT
		// @param allowableResidual 圧力方程式の収束判定（許容誤差）
#endif
		// @param env MPS計算用の計算空間固有パラメータ
		MpsComputer(
#ifndef PRESSURE_EXPLICIT
			const double allowableResidual,
#endif
			const MpsEnvironment& env)
			: environment(env)
		{
#ifndef PRESSURE_EXPLICIT
			// 圧力方程式の許容誤差を設定
			ppe.allowableResidual = allowableResidual;
#endif
		}

		// 時間を進める
		void ForwardTime()
		{
			// 時間刻みを設定
			DetermineDt();

			// 粒子数密度を計算する
			ComputeNeighborDensities();

			// 第一段階の計算
			ComputeExplicitForces();

#ifdef MODIFY_TOO_NEAR
			// 過剰接近粒子の補正
			ModifyTooNear();
#endif

			// 粒子数密度を計算する
			ComputeNeighborDensities();

			// 第二段階の計算
			ComputeImplicitForces();

			// 時間を進める
			environment.SetNextT();
		}

		// 粒子を追加する
		void AddParticle(const Particle& particle)
		{
			particles.push_back(particle);
		}
		void AddParticle(Particle&& particle)
		{
			particles.push_back(std::move(particle));
		}

		// 粒子リストを取得する
		const auto& Particles() const
		{
			return this->particles;
		}

		// 計算空間パラメーターを取得する
		const MpsEnvironment& Environment() const
		{
			return environment;
		}

		// 代入演算子
		// @param src 代入元
		MpsComputer& operator=(const MpsComputer& src)
		{
			this->particles = src.particles;
#ifndef PRESSURE_EXPLICIT
			this->ppe = src.ppe;
#endif
			this->environment = src.environment;
			this->du = src.du;
			return *this;
		}
	};
}
#endif
