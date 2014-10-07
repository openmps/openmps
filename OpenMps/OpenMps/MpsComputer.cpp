#include "MpsComputer.hpp"
#include <algorithm>
#include <cmath>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace OpenMps
{

	MpsComputer::MpsComputer(
#ifndef PRESSURE_EXPLICIT
		const double allowableResidual,
#endif	
		const MpsEnvironment& env,
		const Particle::List& particles)
		: environment(env),
		particles(particles)
	{
#ifndef PRESSURE_EXPLICIT
		// 圧力方程式の許容誤差を設定
		ppe.allowableResidual = allowableResidual;
#endif
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
		environment.SetNextT();
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

		// 時間刻みを設定
		environment.SetDt(maxU);
	}

	void MpsComputer::ComputeNeighborDensities()
	{
		const double r_e = environment.R_e;

		// 全粒子で
		for(auto& particle : particles)
		{
			// 粒子数密度を計算する
			particle->UpdateNeighborDensity(particles, r_e);
		}
	}

	void MpsComputer::ComputeExplicitForces()
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
		for(auto& particle : particles)
		{
			// 重力の計算
			Vector d = g;

			// 粘性項の計算
			auto vis = particle->GetViscosity(particles, n0, r_e, lambda, nu, dt);
			d += vis;
			a.push_back(d);
		}

		// 全粒子で
		for(unsigned int i = 0; i < particles.size(); i++)
		{
			// 位置・速度を修正
			Vector thisA = a[i];
			particles[i]->Move(particles[i]->VectorU() * dt + a[i]*dt*dt/2);
			particles[i]->Accelerate(a[i] * dt);
		}
	}

	void MpsComputer::ComputeImplicitForces()
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
		for(unsigned int i = 0; i < particles.size(); i++)
		{
			const auto n0 = environment.N0();
			const auto surfaceRatio = environment.SurfaceRatio;
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

		// 全粒子で
		for(auto& particle : particles)
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
		for(unsigned int i = 0; i < particles.size(); i++)
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
	void MpsComputer::SetPressurePoissonEquation()
	{
		// 粒子数を取得
		unsigned int n = particles.size();

		// 粒子に増減があれば
		if(n != (int)ppe.b.size())
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
		for(unsigned int i = 0; i < n; i++)
		{
			const auto n0 = environment.N0();
			const auto dt = environment.Dt();
			const auto surfaceRatio = environment.SurfaceRatio;

			// 生成項を計算する
			double b_i = particles[i]->Source(n0, dt, surfaceRatio);
			ppe.b(i) = b_i;

			// 圧力を未知数ベクトルの初期値にする
			double x_i = particles[i]->P();
			ppe.x(i) = x_i;
		}
		// TODO: 以下もそうだけど、圧力方程式を作る際にインデックス指定のfor回さなきゃいけないのが気持ち悪いので、どうにかしたい

		// 係数行列初期化
		ppe.A.clear();

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
					// 自分以外
					if(i != (int)j)
					{
						const auto n0 = environment.N0();
						const auto r_e = environment.R_e;
						const auto rho = environment.Rho;
						const auto lambda = environment.Lambda();
						const auto surfaceRatio = environment.SurfaceRatio;

						// 非対角項を計算
						double a_ij = particles[i]->Matrix(*particles[j], n0, r_e, lambda, rho, surfaceRatio);
						ppe.A(i, j) = a_ij;

						// 対角項も設定
						a_ii -= a_ij;
					}
				}
			}

			// 対角項を設定
			ppe.A(i, i) = a_ii;
		}
	}

	void MpsComputer::SolvePressurePoissonEquation()
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
		for(unsigned int i = 0; (i < x.size())&&(!isConverged); i++)
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
			if(!isConverged)
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

		// 全粒子で
		for(auto& particle : particles)
		{
			const double r_e = environment.R_e;
			const double dt = environment.Dt();
			const double rho = environment.Rho;
			const double n0 = environment.N0();

			// 圧力勾配を計算する
			Vector d = particle->GetPressureGradient(particles, r_e, dt, rho, n0);
			du.push_back(d);
		}

		// 全粒子で
		for(unsigned int i = 0; i < particles.size(); i++)
		{
			const double dt = environment.Dt();

			// 位置・速度を修正
			Vector thisDu = du[i];
			particles[i]->Accelerate(thisDu);
			particles[i]->Move(thisDu * dt);
		}
	}
}
