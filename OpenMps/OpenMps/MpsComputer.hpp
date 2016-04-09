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
#include <viennacl/linalg/inner_prod.hpp>
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
		inline void StoreGrid()
		{
			// TODO: 毎回全消去しない高速化しよう
			grid.Clear();

			// 全粒子について
			for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
			{
				const unsigned int i = static_cast<unsigned int>(ii);

				// グリッドに登録
				grid.AddParticle(particles[i].VectorX(), ii);
			}
		}

		// 時間刻みを決定する
		inline void DetermineDt()
		{
			namespace ublas = boost::numeric::ublas;

			// 最大速度を取得
			auto maxUParticle = *std::max_element(particles.cbegin(), particles.cend(),
				[](const Particle& base, const Particle& target)
			{
				auto baseU = base.VectorU();
				auto targetU = target.VectorU();
				return (ublas::inner_prod(baseU, baseU) < ublas::inner_prod(targetU, targetU));
			});
			auto maxU = ublas::norm_2(maxUParticle.VectorU());

			// 時間刻みを設定
			environment.SetDt(maxU);
		}

		// 粒子数密度を計算する
		inline void ComputeNeighborDensities()
		{
			const double r_e = environment.R_e;

#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
			{
				const unsigned int i = static_cast<unsigned int>(ii);

				// 粒子数密度を計算する
				particles[i].UpdateNeighborDensity(particles, grid, r_e);
			}
		}

		// 陽的にで解く部分（第一段階）を計算する
		inline void ComputeExplicitForces()
		{
			const auto n0 = environment.N0();
			const auto r_e = environment.R_e;
#ifndef MPS_HV
			const auto lambda = environment.Lambda();
#endif
			const auto nu = environment.Nu;
			const auto dt = environment.Dt();
			const auto g = environment.G;

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
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					// 粘性項の計算
					auto vis = particles[i].GetViscosity(particles, grid, n0, r_e,
#ifndef MPS_HV
						lambda,
#endif
						nu);

					// 重力＋粘性項
					a[i] = g + vis;
				}

				// 全粒子で
#ifdef _OPENMP
#pragma omp for
#endif
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					// 位置・速度を修正
					Vector thisA = a[i];
					particles[i].Move(particles[i].VectorU() * dt + a[i] * dt*dt / 2);
					particles[i].Accelerate(a[i] * dt);
				}
			}
		}

		// 陰的にで解く部分（第ニ段階）を計算する
		inline void ComputeImplicitForces()
		{
#ifdef PRESSURE_EXPLICIT
			// 得た圧力を計算する
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
			{
				const unsigned int i = static_cast<unsigned int>(ii);

				const auto c = environment.C;
				const auto n0 = environment.N0();
				const auto rho = environment.Rho;

				particles[i].UpdatePressure(c, rho, n0);
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

				particles[i].P(ppe.x(i), n0, surfaceRatio);
			}
#endif

			// 圧力勾配項を計算する
			ModifyByPressureGradient();
		}

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子を補正する
		inline void ModifyTooNear()
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
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					const auto r_e = environment.R_e;
					const auto rho = environment.Rho;
					const auto tooNearLength = environment.TooNearLength;
					const auto tooNearCoefficient = environment.TooNearCoefficient;

					// 過剰接近粒子からの速度修正量を計算する
					Vector d = particles[i].GetCorrectionByTooNear(particles, grid, r_e, rho, tooNearLength, tooNearCoefficient);
					du[i] = d;
				}

#ifdef _OPENMP
#pragma omp for
#endif
				// 全粒子で
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					const auto dt = environment.Dt();

					// 位置・速度を修正
					Vector thisDu = du[i];
					particles[i].Accelerate(thisDu);
					particles[i].Move(thisDu * dt);
				}
			}
		}
#endif

#ifndef PRESSURE_EXPLICIT
		// 圧力方程式を設定する
		inline void SetPressurePoissonEquation()
		{
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
#ifdef USE_VIENNACL
				ppe.tempA = TempMatrix(n, n);
#endif
#ifdef _OPENMP
				ppe.a_ij.resize(n);
#endif
			}

			// 係数行列初期化
#ifdef USE_VIENNACL
			auto& A = ppe.tempA;
#else
			auto& A = ppe.A;
#endif
			A.clear();

#ifdef _OPENMP
#pragma omp parallel
#endif
			{
#ifdef _OPENMP
#pragma omp for
#endif
				// 全粒子で
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

#ifdef MPS_HS
					const auto r_e = environment.R_e;
#endif
					const auto n0 = environment.N0();
					const auto dt = environment.Dt();
					const auto surfaceRatio = environment.SurfaceRatio;

					// 生成項を計算する
					double b_i = particles[i].GetPpeSource(
#ifdef MPS_HS
						particles, grid, r_e,
#endif
						n0, dt, surfaceRatio);
					ppe.b(i) = b_i;

					// 圧力を未知数ベクトルの初期値にする
					double x_i = particles[i].P();
					ppe.x(i) = x_i;
				}
				// TODO: 以下もそうだけど、圧力方程式を作る際にインデックス指定のfor回さなきゃいけないのが気持ち悪いので、どうにかしたい

#ifdef _OPENMP
#pragma omp for
#endif
				// 全粒子で
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					const auto n0 = environment.N0();
					const auto surfaceRatio = environment.SurfaceRatio;

					// 対角項を初期化
					double a_ii = 0;
#ifdef _OPENMP
					ppe.a_ij[i].clear();
#endif

					if ((particles[i].Type() == Particle::ParticleTypeDummy) || particles[i].IsSurface(n0, surfaceRatio))
					{
						a_ii = 1;
					}
					else
					{
						// 全近傍ブロックで
						for (auto block = grid.cbegin(particles[i].VectorX()); block != grid.cend(particles[i].VectorX()); block++)
						{
							// 近傍ブロック内の粒子を取得
							auto neighborBlock = *block;
							auto neighbors = grid[neighborBlock];

							// 近傍ブロック内の粒子に対して
							for (auto jj : neighbors)
							{
								const unsigned int j = static_cast<unsigned int>(jj);

								const auto r_e = environment.R_e;
								const auto rho = environment.Rho;
#ifndef MPS_HL
								const auto lambda = environment.Lambda();
#endif
								// ダミー粒子と自分以外
								if ((particles[j].Type() != Particle::ParticleTypeDummy) && (i != j))
								{
									// 非対角項を計算
									double a_ij = particles[i].GetPpeMatrix(particles[j], n0, r_e,
#ifndef MPS_HL
										lambda,
#endif
										rho);

									// 自由表面の場合は非対角項は設定しない
									if (!particles[j].IsSurface(n0, surfaceRatio))
									{
#ifdef _OPENMP
										// 非対角項を格納
										ppe.a_ij[i].push_back(Ppe::A_ij(j, a_ij));
#else
										A(i, j) = a_ij;
#endif
									}

									// 対角項も設定
									a_ii -= a_ij;
								}
							}
						}
					}
#ifdef _OPENMP
					// 対角項を格納
					ppe.a_ij[i].push_back(Ppe::A_ij(i, a_ii));
#else
					A(i, i) = a_ii;
#endif
				}
			}

#ifdef _OPENMP
			// 全行の
			for (int ii = 0; ii < static_cast<int>(n); ii++)
			{
				const unsigned int i = static_cast<unsigned int>(ii);

				// 全有効列で
				for (auto k : ppe.a_ij[i])
				{
					// 列番号と値を取得
					const unsigned int j = static_cast<unsigned int>(k.first);
					double a_ij = k.second;

					// 行列に格納
					A(i, j) = a_ij;
				}
			}
#endif

#ifdef USE_VIENNACL
			// 作成した係数行列をデバイス側に複製
			viennacl::copy(ppe.tempA, ppe.A);
#endif
		}

		// 圧力方程式をを解く
		inline void SolvePressurePoissonEquation()
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
				Ap = blas::prod(A, p);
				const double alpha = rr / blas::inner_prod(p, Ap);
				x += alpha * p;
				r -= alpha * Ap;
				const double rrNew = blas::inner_prod(r, r);

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

		// 圧力勾配によって速度と位置を修正する
		inline void ModifyByPressureGradient()
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
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					const double r_e = environment.R_e;
					const double dt = environment.Dt();
					const double rho = environment.Rho;
					const double n0 = environment.N0();

					// 圧力勾配を計算する
					Vector d = particles[i].GetPressureGradient(particles, grid, r_e, dt, rho, n0);
					du[i] = d;
				}
#ifdef _OPENMP
#pragma omp for
#endif

				// 全粒子で
				for (int ii = 0; ii < static_cast<int>(particles.size()); ii++)
				{
					const unsigned int i = static_cast<unsigned int>(ii);

					const double dt = environment.Dt();

					// 位置・速度を修正
					Vector thisDu = du[i];
					particles[i].Accelerate(thisDu);
					particles[i].Move(thisDu * dt);
				}
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
			const MpsEnvironment& env,
			const Particle::List& particles)
			: environment(env),
			grid(env.R_e),
			particles(particles)
		{
#ifndef PRESSURE_EXPLICIT
			// 圧力方程式の許容誤差を設定
			ppe.allowableResidual = allowableResidual;
#endif
		}

		// 時間を進める
		void ForwardTime()
		{
			// グリッドに登録
			StoreGrid();

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

			// グリッドに登録
			StoreGrid();

			// 粒子数密度を計算する
			ComputeNeighborDensities();

			// 第二段階の計算
			ComputeImplicitForces();

			// 時間を進める
			environment.SetNextT();
		}

		// 粒子リストを取得する
		const Particle::List& Particles() const
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
