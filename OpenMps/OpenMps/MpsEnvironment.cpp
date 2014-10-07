#include "MpsEnvironment.hpp"
#include "Particle.hpp"

namespace OpenMps
{
	MpsEnvironment::MpsEnvironment(
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
			const double l_0)
		:t(0), dt(0), G(CreateVector(0, -g)), Rho(rho), Nu(nu), maxDx(courant*l_0), R_e(r_eByl_0 * l_0), SurfaceRatio(surfaceRatio),

#ifdef MODIFY_TOO_NEAR
		TooNearLength(tooNearRatio*l_0), TooNearCoefficient(tooNearCoefficient),
#endif
#ifdef PRESSURE_EXPLICIT
		C(c),

		// 最大時間刻みは、dx < c dt （音速での時間刻み制限）と、指定された引数のうち小さい方
		maxDt(std::min(maxDt, (courant*l_0)/c))
#else
		// 最大時間刻みは、dx < 1/2 g dt^2 （重力による等加速度運動での時間刻み制限）と、指定された引数のうち小さい方
		maxDt(std::min(maxDt, std::sqrt(2*(courant*l_0)/g)))
#endif
	{
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
					auto w = Particle::Weight(r, R_e);

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
}