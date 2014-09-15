#ifndef MPSCOMPUTER_INCLUDED
#define MPSCOMPUTER_INCLUDED
#include "defines.hpp"

#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#ifdef USE_VIENNACL
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/vector.hpp>
#endif
#include "Particle.hpp"

namespace OpenMps
{
	// MPS法による計算空間
	class MpsComputer
	{
	private:
#ifdef USE_VIENNACL
		// 線形方程式用の疎行列
		typedef viennacl::compressed_matrix<double> Matrix;

		// 線形方程式用の多次元ベクトル
		typedef viennacl::vector<double> LongVector;

		// CPU用疎行列（係数行列の設定に使用）
		typedef boost::numeric::ublas::compressed_matrix<double> TempMatrix;
#else
		// 線形方程式用の疎行列
		typedef boost::numeric::ublas::compressed_matrix<double> Matrix;

		// 線形方程式用の多次元ベクトル
		typedef boost::numeric::ublas::vector<double> LongVector;
#endif

		// 粒子リスト
		Particle::List particles;

		// 現在時刻
		double t;

		// 時間刻み
		double dt;

		// 基準粒子数密度
		double n0;

		// 重力加速度
		Vector g;

		// 密度
		double rho;

		// 動粘性係数
		double nu;

#ifdef PRESSURE_EXPLICIT
		// 音速
		double c;
#endif

		// 1ステップの最大移動距離
		double maxDx;

		// 最大時間刻み
		double maxDt;

		// 影響半径
		double r_e;

		// 自由表面を判定する係数
		double surfaceRatio;

		// 拡散モデル定数
		double lambda;

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子と判定される距離
		double tooNearLength;

		// 過剰接近粒子から受ける修正量の係数
		double tooNearCoefficient;
#endif

#ifndef PRESSURE_EXPLICIT
		// 圧力方程式
		struct Ppe
		{
			// 係数行列
			Matrix A;

#ifdef USE_VIENNACL
			// CPU用疎行列（係数行列の設定に使用）
			TempMatrix tempA;
#endif

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
		inline void DetermineDt();

		// 粒子数密度を計算する
		inline void ComputeNeighborDensities();

		// 陽的にで解く部分（第一段階）を計算する
		inline void ComputeExplicitForces();

		// 陰的にで解く部分（第ニ段階）を計算する
		inline void ComputeImplicitForces();

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子を補正する
		inline void ModifyTooNear();
#endif
		
#ifndef PRESSURE_EXPLICIT
		// 圧力方程式を設定する
		inline void SetPressurePoissonEquation();

		// 圧力方程式をを解く
		inline void SolvePressurePoissonEquation();
#endif

		// 圧力勾配によって速度と位置を修正する
		inline void ModifyByPressureGradient();
		
	public:
		struct Exception
		{
			std::string Message;
		};
		
		// @param maxDt 最大時間刻み（出力時間刻み以下など）
		// @param g 重力加速度
		// @param c 音速
		// @param rho 密度
		// @param nu 動粘性係数
		// @param C クーラン数
		// @param r_eByl_0 影響半径と初期粒子間距離の比
		// @param surfaceRatio 自由表面判定の係数
		// @param allowableResidual 圧力方程式の収束判定（許容誤差）
		// @param l_0 初期粒子間距離
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		MpsComputer(
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
			const double& l_0);

		// 時間を進める
		void ForwardTime();

		// 粒子を追加する
		// @param particle 追加する粒子
		void AddParticle(const Particle& particle);

		// 粒子リストを取得する
		inline const Particle::List Particles() const
		{
			return this->particles;
		}

		// 現在時刻を取得する
		inline double T() const
		{
			return t;
		}
	};
}
#endif