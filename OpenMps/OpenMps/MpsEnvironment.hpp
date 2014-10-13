#ifndef MPSENVIRONMENT_INCLUDED
#define MPSENVIRONMENT_INCLUDED

#include "Vector.hpp"

namespace OpenMps
{
	// MPS計算用の計算空間固有パラメータ
	class MpsEnvironment
	{
	private:
		// 現在時刻
		double t;

		// 時間刻み
		double dt;

		// 基準粒子数密度
		double n0;

		// 拡散モデル定数
		double lambda;

		// 最大時間刻み
		const double maxDt;

		// 1ステップの最大移動距離
		const double maxDx;

	public:

		// 影響半径
		const double R_e;

		// 重力加速度
		const Vector G;

		// 密度
		const double Rho;

		// 動粘性係数
		const double Nu;

		// 自由表面を判定する係数
		const double SurfaceRatio;

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子と判定される距離
		const double TooNearLength;

		// 過剰接近粒子から受ける修正量の係数
		const double TooNearCoefficient;
#endif

#ifdef PRESSURE_EXPLICIT
		// 音速
		const double C;
#endif

		// @param maxDt 最大時間刻み（出力時間刻み以下など）
		// @param courant クーラン数
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		// @param g 重力加速度
		// @param rho 密度
		// @param nu 動粘性係数
		// @param r_eByl_0 影響半径と初期粒子間距離の比
		// @param surfaceRatio 自由表面判定の係数
		// @param c 音速
		// @param l_0 初期粒子間距離
		MpsEnvironment(
			const double maxDt,
			const double courant,
#ifdef MODIFY_TOO_NEAR
			const double tooNearRatio,
			const double tooNearCoefficient,
#endif
			const double g,
			const double rho,
			const double nu,
			const double surfaceRatio,
			const double r_eByl_0,
#ifdef PRESSURE_EXPLICIT
			const double c,
#endif
			const double l_0);

		// する
		inline void SetDt(const double maxU)
		{
			dt = std::min(maxDx/maxU, maxDt);
		}
		// CFL条件より時間刻みを決定する

		// 時刻を進める
		inline void SetNextT()
		{
			t += dt;
		}

		inline double T() const
		{
			return t;
		}

		inline double Dt() const
		{
			return dt;
		}

		inline double N0() const
		{
			return n0;
		}

		inline double Lambda() const
		{
			return lambda;
		}

		// 代入演算子
		// @param src 代入元
		MpsEnvironment& operator=(const MpsEnvironment& src)
		{	
			this->t = src.t;
			this->dt = src.dt;
			this->n0 = src.n0;
			this->lambda = src.lambda;
			const_cast<double&>(this->maxDt) = src.maxDt;
			const_cast<double&>(this->maxDx) = src.maxDx;
			const_cast<double&>(this->R_e) = src.R_e;
			const_cast<Vector&>(this->G) = src.G;
			const_cast<double&>(this->Rho) = src.Rho;
			const_cast<double&>(this->Nu) = src.Nu;
			const_cast<double&>(this->SurfaceRatio) = src.SurfaceRatio;
#ifdef MODIFY_TOO_NEAR
			const_cast<double&>(TooNearLength) = src.TooNearLength;
			const_cast<double&>(TooNearCoefficient) = src.TooNearCoefficient;
#endif
#ifdef PRESSURE_EXPLICIT
			const_cast<double&>(C) = src.C;
#endif
			return *this;
		}
	};
}
#endif