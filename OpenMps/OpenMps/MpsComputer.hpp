#ifndef MPSCOMPUTER_INCLUDED
#define MPSCOMPUTER_INCLUDED
#include "defines.hpp"

#pragma warning(push, 0)
#include <vector>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#endif
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#ifdef USE_VIENNACL
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/vector.hpp>
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#pragma warning(pop)

#include "Particle.hpp"
#include "Grid.hpp"
#include "MpsEnvironment.hpp"

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

		// 計算空間のパラメーター
		MpsEnvironment environment;

		// 近傍粒子探索用グリッド
		Grid grid;
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

#ifdef _OPENMP
		// 行列成分
		typedef std::pair<int, double> A_ij;

		typedef std::vector< std::vector<A_ij> > A_ijList;

		// 各行の成分
		A_ijList a_ij;
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

		// グリッドに登録する
		inline void StoreGrid();

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
		const MpsEnvironment Environment() const
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
