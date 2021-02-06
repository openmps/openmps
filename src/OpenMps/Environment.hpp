#ifndef ENVIRONMENT_INCLUDED
#define ENVIRONMENT_INCLUDED

#include "Vector.hpp"

namespace { namespace OpenMps
{
	// MPS計算用の計算空間固有パラメータ
	class Environment final
	{
	private:
		// 現在時刻
		double t;

		// 時間刻み
		double dt;

		// 基準粒子数密度
		double n0;

#ifndef MPS_HL
		// 拡散モデル定数
		double lambda;
#endif

	public:

#ifdef PRESSURE_EXPLICIT
		// 音速
		const double C;
#endif

#ifdef ARTIFICIAL_COLLISION_FORCE
		// 過剰接近粒子と判定される距離
		const double TooNearLength;

		// 過剰接近粒子から受ける修正量の係数
		const double TooNearCoefficient;
#endif

		// 最大時間刻み
		const double MaxDt;

		// 1ステップの最大移動距離
		const double MaxDx;

		// 初期粒子間距離
		const double L_0;

		// 影響半径
		const double R_e;

		// 重力加速度
		const Vector G;

		// 密度
		const double Rho;

		// 動粘性係数
		const double Nu;

#ifndef MPS_SPP
		// 自由表面を判定する係数
		const double SurfaceRatio;
#endif

		// 計算空間の最小座標
		const Vector MinX;

		// 計算空間の最大座標
		const Vector MaxX;

		// 近傍粒子として保持する距離
		const double NeighborLength;

		// @param maxDt 最大時間刻み（出力時間刻み以下など）
		// @param courant クーラン数
#ifdef ARTIFICIAL_COLLISION_FORCE
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
#endif
		// @param g 重力加速度
		// @param rho 密度
		// @param nu 動粘性係数
		// @param r_eByl_0 影響半径と初期粒子間距離の比
#ifndef MPS_SPP
		// @param surfaceRatio 自由表面判定の係数
#endif
		// @param c 音速
		// @param l_0 初期粒子間距離
		// @param minX 計算空間の最小X座標
#ifdef DIM3
		// @param minY 計算空間の最小Y座標
#endif
		// @param minZ 計算空間の最小Z座標
		// @param maxX 計算空間の最大X座標
#ifdef DIM3
		// @param maxY 計算空間の最大Y座標
#endif
		// @param maxZ 計算空間の最大Z座標
		Environment(
			const double maxDt,
			const double courant,
#ifdef ARTIFICIAL_COLLISION_FORCE
			const double tooNearRatio,
			const double tooNearCoefficient,
#endif
			const double g,
			const double rho,
			const double nu,
#ifndef MPS_SPP
			const double surfaceRatio,
#endif
			const double r_eByl_0,
#ifdef PRESSURE_EXPLICIT
			const double c,
#endif
			const double l_0,
			const double minX,
#ifdef DIM3
			const double minY,
#endif
			const double minZ,
			const double maxX,
#ifdef DIM3
			const double maxY,
#endif
			const double maxZ)
			:t(0), dt(0),

#ifdef PRESSURE_EXPLICIT
			C(c),

			// 最大時間刻みは、dx < c dt （音速での時間刻み制限）と、指定された引数のうち小さい方
			MaxDt(std::min(maxDt, (courant*l_0) / c)),
#else
			// 最大時間刻みは、dx < 1/2 g dt^2 （重力による等加速度運動での時間刻み制限）と、指定された引数のうち小さい方
			MaxDt(std::min(maxDt, std::sqrt(2 * (courant*l_0) / g))),
#endif
#ifdef ARTIFICIAL_COLLISION_FORCE
			TooNearLength(tooNearRatio*l_0), TooNearCoefficient(tooNearCoefficient),
#endif
			MaxDx(courant*l_0),
			L_0(l_0),
			R_e(r_eByl_0 * l_0),
#ifdef DIM3
			G(CreateVector(0, 0, -g)),
#else
			G(CreateVector(0, -g)),
#endif
			Rho(rho),
			Nu(nu),
#ifndef MPS_SPP
			SurfaceRatio(surfaceRatio),
#endif
#ifdef DIM3
			MinX(CreateVector(minX, minY, minZ)), MaxX(CreateVector(maxX, maxY, maxZ)),
#else
			MinX(CreateVector(minX, minZ)), MaxX(CreateVector(maxX, maxZ)),
#endif
			NeighborLength(r_eByl_0 * l_0 * (1 + courant*2)) // 計算の安定化のためクーラン数の2倍の距離までを近傍粒子として保持する
		{
			// 基準粒子数密度とλの計算
			const auto range = static_cast<int>(std::ceil(r_eByl_0));
			n0 = 0;
#ifndef MPS_HL
			lambda = 0;
#endif
			for (auto i = -range; i < range; i++)
			{
				for (auto j = -range; j < range; j++)
				{
#ifdef DIM3
					for (auto k = -range; k < range; k++)
					{
						// 自分以外との
						if (!((i == 0) && (j == 0) && (k == 0)))
						{
							// 相対位置を計算
							const auto x = CreateVector(i*l_0, j*l_0, k*l_0);
#else
						// 自分以外との
						if (!((i == 0) && (j == 0)))
						{
							// 相対位置を計算
							const auto x = CreateVector(i*l_0, j*l_0);
#endif

							// 影響半径内なら
							const auto r = boost::numeric::ublas::norm_2(x);
							if (r < R_e)
							{
								// 重み関数を計算
								const auto w = Particle::W(r, R_e);

								// 基準粒子数密度に足す
								n0 += w;

#ifndef MPS_HL
								// λに足す
								lambda += r*r * w;
#endif
							}
						}
#ifdef DIM3
					}
#endif
				}
			}

#ifndef MPS_HL
			// λの最終計算
			// λ = (Σr^2 w)/(Σw)
			lambda /= n0;
#endif
		}

		Environment(Environment&&) = default;
		Environment(const Environment&) = default;

		// 時刻を進める
		void SetNextT()
		{
			t += dt;
		}

		double T() const
		{
			return t;
		}

		double& Dt()
		{
			return dt;
		}
		double Dt() const
		{
			return dt;
		}

		double N0() const
		{
			return n0;
		}

#ifndef MPS_HL
		double Lambda() const
		{
			return lambda;
		}
#endif

	};
}}
#endif
