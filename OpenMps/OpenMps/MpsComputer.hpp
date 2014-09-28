﻿#ifndef MPSCOMPUTER_INCLUDED
#define MPSCOMPUTER_INCLUDED

#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
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
		Particle::List particles;

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


		// 時間刻みを決定する
		void DetermineDt();

		// 粒子数密度を計算する
		void ComputeNeighborDensities();

		// 陽的にで解く部分（第一段階）を計算する
		void ComputeExplicitForces();

		// 陰的にで解く部分（第ニ段階）を計算する
		void ComputeImplicitForces();

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子を補正する
		void ModifyTooNear();
#endif
		
#ifndef PRESSURE_EXPLICIT
		// 圧力方程式を設定する
		void SetPressurePoissonEquation();

		// 圧力方程式をを解く
		void SolvePressurePoissonEquation();
#endif

		// 圧力勾配によって速度と位置を修正する
		void ModifyByPressureGradient();
		
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
			const MpsEnvironment& env,
			const Particle::List& particles);

		// 時間を進める
		void ForwardTime();

		// 粒子リストを取得する
		const Particle::List Particles() const
		{
			return this->particles;
		}

		// 計算空間パラメーターを取得する
		inline const MpsEnvironment Environment() const
		{
			return environment;
		}
	};
}
#endif