#include "MpsComputer.hpp"
#include <algorithm>
#include <cmath>
#ifdef USE_VIENNACL
#include <viennacl/linalg/prod.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#endif

namespace OpenMps
{

	MpsComputer::MpsComputer(
			const double& maxDt,
			const double& g,
			const double& rho,
			const double& nu,
			const double& C,
			const double& r_eByl_0,
			const double& surfaceRatio,
#ifdef PRESSURE_EXPLICIT
			const double& c,
#else
			const double& allowableResidual,
#endif
#ifdef MODIFY_TOO_NEAR
			const double& tooNearRatio,
			const double& tooNearCoefficient,
#endif
			const double& l_0)
		: t(0), dt(0), g(), rho(rho), nu(nu), maxDx(C*l_0), r_e(r_eByl_0 * l_0), surfaceRatio(surfaceRatio),
#ifdef MODIFY_TOO_NEAR
		tooNearLength(tooNearRatio*l_0), tooNearCoefficient(tooNearCoefficient),
#endif
#ifdef PRESSURE_EXPLICIT
		c(c),

		// 最大時間刻みは、dx < c dt （音速での時間刻み制限）と、指定された引数のうち小さい方
		maxDt(std::min(maxDt, maxDx/c))
#else
		// 最大時間刻みは、dx < 1/2 g dt^2 （重力による等加速度運動での時間刻み制限）と、指定された引数のうち小さい方
		maxDt(std::min(maxDt, std::sqrt(2*maxDx/g)))
#endif
	{
		// 重力加速度を設定
		this->g[0] = 0;
		this->g[1] = -g;

#ifndef PRESSURE_EXPLICIT
		// 圧力方程式の許容誤差を設定
		ppe.allowableResidual = allowableResidual;
#endif

		// 基準粒子数密度とλの計算
		int range = (int)std::ceil(r_eByl_0);
		n0 = 0;
		lambda = 0;
		for(int i = -range; i < range; i++)
		{
			for(int j = -range; j < range; j++)
			{
				// 自分以外との
				if(!((i == 0) && (j == 0)))
				{
					// 相対位置を計算
					Vector x;
					x[0] = i*l_0;
					x[1] = j*l_0;

					// 重み関数を計算	
					double r = boost::numeric::ublas::norm_2(x);
					auto w = Particle::Weight(r, r_e);

					// 基準粒子数密度に足す
					n0 += w;

					// λに足す
					lambda += r*r * w;
				}
			}
		}

		// λの最終計算
		// λ = (Σr^2 w)/(Σw)
		lambda /= n0;
	}

	void MpsComputer::ForwardTime()
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
		t += dt;
	}

	void MpsComputer::DetermineDt()
	{
		namespace ublas = boost::numeric::ublas;

		// 最大速度を取得
		auto maxUParticle = *std::max_element(particles.cbegin(), particles.cend(),
			[](const Particle::Ptr& base, const Particle::Ptr& target)
			{
				auto baseU = base->VectorU();
				auto targetU = target->VectorU();
				return ( ublas::inner_prod(baseU, baseU) < ublas::inner_prod(targetU, targetU));
			});
		auto maxU = ublas::norm_2(maxUParticle->VectorU());

		// CFL条件より時間刻みを決定
		dt = std::min(maxDx/maxU, maxDt);
	}

	void MpsComputer::ComputeNeighborDensities()
	{
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		for(int i = 0; i < (int)particles.size(); i++)
		{
			// 粒子数密度を計算する
			particles[i]->UpdateNeighborDensity(particles, r_e);
		}
	}

	void MpsComputer::ComputeExplicitForces()
	{
		// 加速度を全初期化
		auto& a = du;
		a.clear();
		a.resize(particles.size());

#ifdef _OPENMP
		#pragma omp parallel
#endif
		{
#ifdef _OPENMP
			#pragma omp for
#endif
			// 全粒子で
			for(int i = 0; i < (int)particles.size(); i++)
			{
				// 粘性項の計算
				auto vis = particles[i]->GetViscosity(particles, n0, r_e, lambda, nu, dt);

				// 重力＋粘性項
				a[i] = g + vis;
			}

			// 全粒子で
#ifdef _OPENMP
			#pragma omp for
#endif
			for(int i = 0; i < (int)particles.size(); i++)
			{
				// 位置・速度を修正
				Vector thisA = a[i];
				particles[i]->Move(particles[i]->VectorU() * dt + a[i]*dt*dt/2);
				particles[i]->Accelerate(a[i] * dt);
			}
		}
	}

	void MpsComputer::ComputeImplicitForces()
	{
#ifdef PRESSURE_EXPLICIT
		// 得た圧力を計算する
#ifdef _OPENMP
		#pragma omp parallel for
#endif
		for(int i = 0; i < (int)particles.size(); i++)
		{
			particles[i]->UpdatePressure(c, rho, n0);
		}
#else
		// 圧力方程式を設定
		SetPressurePoissonEquation();

		// 圧力方程式を解く
		SolvePressurePoissonEquation();

		// 得た圧力を代入する
		for(unsigned int i = 0; i < particles.size(); i++)
		{
			particles[i]->P(ppe.x(i), n0, surfaceRatio);
		}
#endif

		// 圧力勾配項を計算する
		ModifyByPressureGradient();
	}

#ifdef MODIFY_TOO_NEAR
	void MpsComputer::ModifyTooNear()
	{
		// 速度修正量を全初期化
		du.clear();
		du.resize(particles.size());

#ifdef _OPENMP
		#pragma omp parallel
#endif
		{
#ifdef _OPENMP
			#pragma omp for
#endif
			// 全粒子で
			for(int i = 0; i < (int)particles.size(); i++)
			{
				// 過剰接近粒子からの速度修正量を計算する
				Vector d = particles[i]->GetCorrectionByTooNear(particles, r_e, rho, tooNearLength, tooNearCoefficient);
				du[i] = d;
			}

#ifdef _OPENMP
			#pragma omp for
#endif
			// 全粒子で
			for(int i = 0; i < (int)particles.size(); i++)
			{
				// 位置・速度を修正
				Vector thisDu = du[i];
				particles[i]->Accelerate(thisDu);
				particles[i]->Move(thisDu * dt);
			}
		}
	}
#endif

#ifndef PRESSURE_EXPLICIT
	void MpsComputer::SetPressurePoissonEquation()
	{
		// 粒子数を取得
		unsigned int n = particles.size();

		// 粒子に増減があれば
		if(n != ppe.b.size())
		{
			// サイズを変えて作り直し
			ppe.A = Matrix(n, n);
			ppe.x = LongVector(n);
			ppe.b = LongVector(n);
			ppe.cg.r = LongVector(n);
			ppe.cg.p = LongVector(n);
			ppe.cg.Ap = LongVector(n);	
#ifdef USE_VIENNACL
			ppe.tempA = TempMatrix(n, n);
#endif
		}
		// 全粒子で
		for(unsigned int i = 0; i < n; i++)
		{
			// 生成項を計算する
			double b_i = particles[i]->Source(n0, dt, surfaceRatio);
			ppe.b(i) = b_i;

			// 圧力を未知数ベクトルの初期値にする
			double x_i = particles[i]->P();
			ppe.x(i) = x_i;
		}
		// TODO: 以下もそうだけど、圧力方程式を作る際にインデックス指定のfor回さなきゃいけないのが気持ち悪いので、どうにかしたい

		// 係数行列初期化
#ifdef USE_VIENNACL
		auto& A = ppe.tempA;
#else
		auto& A = ppe.A;
#endif
		A.clear();

		// 全粒子で
		for(unsigned int i = 0; i < n; i++)
		{
			// 対角項を初期化
			double a_ii = 0;
			
			// 他の粒子に対して
			// TODO: 全粒子探索してるので遅い
			for(unsigned int j = 0; j < particles.size(); j++)
			{
				// 自分以外
				if(i != j)
				{
					// 非対角項を計算
					double a_ij = particles[i]->Matrix(*particles[j], n0, r_e, lambda, rho, surfaceRatio);
					if(a_ij != 0)
					{
						A(i, j) = a_ij;

						// 対角項も設定
						a_ii -= a_ij;
					}
				}
			}

			// 対角項を設定
			A(i, i) = a_ii;
		}
		
#ifdef USE_VIENNACL
		// 作成した係数行列をデバイス側に複製
		viennacl::copy(ppe.tempA, ppe.A);
#endif
	}

	void MpsComputer::SolvePressurePoissonEquation()
	{
		// 共役勾配法で解く
		// TODO: 前処理ぐらい入れようよ

		auto& A = ppe.A;
		auto& x = ppe.x;
		auto& b = ppe.b;
		auto& r = ppe.cg.r;
		auto& p = ppe.cg.p;
		auto& Ap = ppe.cg.Ap;

		// 使用する演算を選択
#ifdef USE_VIENNACL
		namespace blas = viennacl::linalg;
#else
		namespace blas = boost::numeric::ublas;
#endif

		// 初期値を設定
		//  (Ap)_0 = A * x
		//  r_0 = b - Ap
		//  p_0 = r_0
		//  rr = r・r
		Ap = blas::prod(A, x);
		r = b - Ap;
		p = r;
		double rr = blas::inner_prod(r, r);

		// 初期値でまだ収束していない場合
		bool isConverged = (blas::norm_inf(r)< ppe.allowableResidual);
		if(!isConverged)
		{
			// 未知数分だけ繰り返す
			for(unsigned int i = 0; i < x.size(); i++)
			{
				// 計算を実行
				//  Ap = A * p
				//  α = rr/(p・Ap)
				//  x' += αp
				//  r' -= αAp
				Ap = blas::prod(A, p);
				double alpha = rr / blas::inner_prod(p, Ap);
				x += alpha * p;
				r -= alpha * Ap;

				// 収束判定（残差＝残差ベクトルの最大要素値）
				double residual = blas::norm_inf(r);
				isConverged = (residual < ppe.allowableResidual);

				// 収束していたら
				if(isConverged)
				{
					// 繰り返し終了
					break;
				}
				// なかったら
				else
				{
					// 残りの計算を実行
					//  r'r' = r'・r'
					//  β= r'r'/rr
					//  p = r' + βp
					//  rr = r'r'
					double rrNew = blas::inner_prod(r, r);
					double beta = rrNew / rr;
					p = r + beta * p;
					rr = rrNew;
				}
			}
		}

		// 理論上は未知数分だけ繰り返せば収束するはずだが、収束しなかった場合は
		if(!isConverged)
		{
			// どうしようもないので例外
			Exception exception;
			exception.Message = "Conjugate Gradient method couldn't solve Pressure Poison Equation";
			throw exception;
		}
	};
#endif

	void MpsComputer::ModifyByPressureGradient()
	{
		// 速度修正量を全初期化
		du.clear();
		du.resize(particles.size());

#ifdef _OPENMP
		#pragma omp parallel
#endif
		{
#ifdef _OPENMP
			#pragma omp for
#endif
			// 全粒子で
			for(int i = 0; i < (int)particles.size(); i++)
			{
				// 圧力勾配を計算する
				Vector d = particles[i]->GetPressureGradient(particles, r_e, dt, rho, n0);
				du[i] = d;
			}
#ifdef _OPENMP
			#pragma omp for
#endif

			// 全粒子で
			for(int i = 0; i < (int)particles.size(); i++)
			{
				// 位置・速度を修正
				Vector thisDu = du[i];
				particles[i]->Accelerate(thisDu);
				particles[i]->Move(thisDu * dt);
			}
		}
	}

	void MpsComputer::AddParticle(const Particle::Ptr& particle)
	{
		particles.push_back(particle);
	}
}