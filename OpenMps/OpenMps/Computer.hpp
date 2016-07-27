#ifndef COMPUTER_INCLUDED
#define COMPUTER_INCLUDED
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
#include "Environment.hpp"
#include "Grid.hpp"

namespace OpenMps
{
	namespace Detail
	{
		namespace Field
		{
			enum class Name
			{
				ID,
				X,
				U,
				P,
				N,
				Type,
			};

			template<typename T, typename PARTICLES>
			using Getter = T (*)(const PARTICLES& particles, const std::size_t i);

			template<Name NAME>
			struct GetGetter;
			template<>
			struct GetGetter<Name::ID> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES&, const std::size_t i)
				{
					return i;
				}
			};
			template<>
			struct GetGetter<Name::X> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES& particles, const std::size_t i)
				{
					return particles[i].X();
				}
			};
			template<>
			struct GetGetter<Name::U> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES& particles, const std::size_t i)
				{
					return particles[i].U();
				}
			};
			template<>
			struct GetGetter<Name::P> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES& particles, const std::size_t i)
				{
					return particles[i].P();
				}
			};
			template<>
			struct GetGetter<Name::N> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES& particles, const std::size_t i)
				{
					return particles[i].N();
				}
			};
			template<>
			struct GetGetter<Name::Type> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES& particles, const std::size_t i)
				{
					return particles[i].TYPE();
				}
			};

			template<typename PARTICLES, Name NAME>
			constexpr auto GetGetters()
			{
				return std::make_tuple(&GetGetter<NAME>::Get<PARTICLES>);
			}

			template<typename PARTICLES, Name NAME0, Name NAME1, Name... NAMES>
			constexpr auto GetGetters()
			{
				return std::tuple_cat(
					GetGetters<PARTICLES, NAME0>(),
					GetGetters<PARTICLES, NAME1, NAMES...>());
			}


			template<typename TUPLE, std::size_t I = 0, bool IS_END = (I == std::tuple_size<TUPLE>::value)>
			struct GetArg;

			template<typename TUPLE, std::size_t I>
			struct GetArg<TUPLE, I, true> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES&, const std::size_t, TUPLE)
				{
					return std::make_tuple();
				}
			};
			template<typename TUPLE, std::size_t I>
			struct GetArg<TUPLE, I, false> final
			{
				template<typename PARTICLES>
				static auto Get(const PARTICLES& particles, const std::size_t i, TUPLE getters)
				{
					return std::tuple_cat(
						std::make_tuple(std::get<I>(getters)(particles, i)),
						GetArg<TUPLE, I+1>::Get(particles, i, getters));
				}
			};

			template<typename PARTICLES, typename... Ts>
			auto Get(const PARTICLES& particles, const std::size_t i, std::tuple<Getter<Ts, PARTICLES>...> getters)
			{
				return GetArg<decltype(getters)>::Get(particles, i, getters);
			}
		}

		template<typename FUNC, typename TUPLE, std::size_t I = 0, bool IS_END = (I == std::tuple_size<TUPLE>::value)>
		struct Invoker;

		template<typename FUNC, typename TUPLE, std::size_t I>
		struct Invoker<FUNC, TUPLE, I, true> final
		{
			template<typename... ARGS>
			static auto Invoke(TUPLE, FUNC func, ARGS... args)
			{
				return func(std::forward<ARGS>(args)...);
			}
		};
		template<typename FUNC, typename TUPLE, std::size_t I>
		struct Invoker<FUNC, TUPLE, I, false> final
		{
			template<typename... ARGS>
			static auto Invoke(TUPLE tuple, FUNC func, ARGS... args)
			{
				return Invoker<FUNC, TUPLE, I + 1>::Invoke(tuple, func, std::forward<ARGS>(args)..., std::get<I>(tuple));
			}

			static auto Invoke(TUPLE tuple, FUNC func)
			{
				return Invoker<FUNC, TUPLE, I + 1>::Invoke(tuple, func, std::get<I>(tuple));
			}
		};

		template<typename FUNC, typename... ARGS>
		auto Invoke(std::tuple<ARGS...> tuple, FUNC func)
		{
			return Invoker<FUNC, decltype(tuple)>::Invoke(tuple, func);
		}
	}

	// MPS法による計算空間
	class Computer final
	{
	private:
		// 粒子リスト
		std::vector<Particle> particles;

		// 計算空間のパラメーター
		Environment environment;

		// 近傍粒子探索用のグリッド
		Grid grid;

		// 近傍粒子リスト
		boost::multi_array<std::size_t, 2> neighbor;

#ifndef PRESSURE_EXPLICIT
		// 圧力方程式
		struct Ppe
		{
			// 線形方程式用の疎行列
			using Matrix = boost::numeric::ublas::compressed_matrix<double>;

			// 線形方程式用の多次元ベクトル
			using LongVector = boost::numeric::ublas::vector<double>;

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

#ifdef MPS_ECS
		// 誤差修正量
		std::vector<double> ecs;
#endif

		// 2点間の距離を計算する
		// @param x1 点1
		// @param x2 点2
		static double R(const Vector& x1, const Vector& x2)
		{
			const auto r = x1 - x2;
			return std::sqrt(boost::numeric::ublas::inner_prod(r, r));
		}
		// 2粒子間の距離を計算する
		// @param p1 粒子1
		// @param p2 粒子2
		static double R(const Particle& p1, const Particle& p2)
		{
			return R(p1.X(), p2.X());
		}

		// 自由表面かどうかの判定
		// @param n 粒子数密度
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		static bool IsSurface(const double n, const double n0, const double surfaceRatio)
		{
			return n / n0 < surfaceRatio;
		}

		// 近傍粒子との相互作用を計算する
		// @tparam FIELDS 必要な粒子の物理量
		// @param zero 初期値
		// @param func 相互作用関数
		template<Detail::Field::Name... FIELDS, typename FUNC, typename ZERO>
		auto AccumulateNeighbor(const std::size_t i, ZERO&& zero, const FUNC func) const
		{
			// VS2015 Update2だとなぜか定数式だと評価できないと言われるので・・・
			// c.f. https://connect.microsoft.com/VisualStudio/feedback/details/2599450
#if _MSC_FULL_VER == 190023918
			const
#else
			constexpr
#endif
				auto getter = Detail::Field::GetGetters<decltype(particles), FIELDS...>();

			auto sum = std::move(zero);

			// 他の粒子に対して
			const auto n = NeighborCount(i);
			for(auto idx = decltype(n)(0); idx < n; idx++)
			{
				const auto j = Neighbor(i, idx);

				// 自分は近傍粒子に含まない
				if(j != i)
				{
					// 近傍粒子を生成する時に無効粒子と自分自身は除外されているので特になにもしない
					sum += Detail::Invoke(Detail::Field::Get(particles, j, getter), func);
				}
			}

			return sum;
		};

		// 近傍粒子数
		// @param i 対象の粒子番号
		auto& NeighborCount(const std::size_t i)
		{
			return neighbor[static_cast<decltype(neighbor)::index>(i)][0]; // 各行の先頭が近傍粒子数
		}
		auto NeighborCount(const std::size_t i) const
		{
			return neighbor[static_cast<decltype(neighbor)::index>(i)][0]; // 各行の先頭が近傍粒子数
		}

		// 近傍粒子番号
		// @param i 対象の粒子番号
		auto& Neighbor(const std::size_t i, const std::size_t idx)
		{
			return neighbor[static_cast<decltype(neighbor)::index>(i)][1 + static_cast<decltype(neighbor)::index>(idx)]; // 各行の先頭は近傍粒子数なので
		}
		auto Neighbor(const std::size_t i, const std::size_t idx) const
		{
			return neighbor[static_cast<decltype(neighbor)::index>(i)][1 + static_cast<decltype(neighbor)::index>(idx)]; // 各行の先頭は近傍粒子数なので
		}

		// 近傍粒子探索
		void SearchNeighbor()
		{
			// 全消去（TODO: 全消去じゃない形に）
			grid.Clear();

			// グリッドに格納
			const auto n = particles.size();
			for(auto i = decltype(n)(0); i < n; i++)
			{
				// 無効粒子は格納しない
				if(particles[i].TYPE() != Particle::Type::Disabled)
				{
					// 格納に失敗した時は領域外なので無効化する
					const bool ok = grid.Store(particles[i].X(), i);
					if(!ok)
					{
						particles[i].Disable();
					}
				}
			}

			// 近傍粒子を格納
			for(auto i = decltype(n)(0); i < n; i++)
			{
				// 無効粒子は除く
				if(particles[i].TYPE() != Particle::Type::Disabled)
				{
					auto idx = decltype(i)(0);

					const auto&& begin = grid.cbegin(particles[i].X());
					const auto&& end = grid.cend();
					for(auto it = std::move(begin); !(it == end); ++it)
					{
						const auto j = *it;

						// 自分自身と無効粒子は除外
						if((j != i) && (particles[j].TYPE() != Particle::Type::Disabled))
						{
							// 半径内なら近傍粒子とする
							const auto r = R(particles[i], particles[j]);
							if(r < environment.NeighborLength)
							{
								Neighbor(i, idx) = j;
								idx++;
							}
						}
					}

					NeighborCount(i) = idx;
				}
			}
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
			const auto n = particles.size();
			for(auto i = decltype(n)(0); i < n; i++)
			{
				auto& particle = particles[i];

				// ダミー粒子と無効粒子を除く
				if((particle.TYPE() != Particle::Type::Dummy) && (particle.TYPE() != Particle::Type::Disabled))
				{
					// 粒子数密度を計算する
					particle.N() = AccumulateNeighbor<Detail::Field::Name::X>(i, 0.0, [&thisX = particle.X(), &r_e](const Vector& x)
					{
						return Particle::W(R(thisX, x), r_e);
					});
				}
			}
		}

#ifndef PRESSURE_EXPLICIT
		// 粒子数密度の瞬間増加速度(Dn/Dt)を計算する
		// @param i 対象の粒子番号
		auto NeighborDensitiyVariationSpeed(const std::size_t i)
		{
#ifdef MPS_HS
			const auto r_e = environment.R_e;

			// HS法（高精度生成項）：-r_eΣ r・u / |r|^3
			const auto result = -r_e * AccumulateNeighbor<Detail::Field::Name::X, Detail::Field::Name::U, Detail::Field::Name::Type>(i, 0.0,
				[&thisX = particles[i].X(), &thisU = particles[i].U()](const Vector& x, const Vector& u, const Particle::Type type)
			{
				// ダミー粒子以外
				if(type != Particle::Type::Dummy)
				{
					const Vector dx = x - thisX;
					const Vector du = u - thisU;
					const auto r = boost::numeric::ublas::norm_2(dx);
					return boost::numeric::ublas::inner_prod(dx, du) / (r*r*r);
				}
				else
				{
					return 0.0;
				}
			});
#else
			const auto n0 = environment.N0();
			// 標準MPS法：b_i = (n_i - n0)/Δt
			const auto result = (thisN - n0) / dt;
#endif
			return result;
		}
#endif

#ifdef MPS_ECS
		// 誤差修正量を計算する
		void ComputeErrorCorrection()
		{
			const auto n0 = environment.N0();

			// 全粒子で
			const auto n = particles.size();
			for(auto i = decltype(n)(0); i < n; i++)
			{
				const auto& particle = particles[i];

				// ダミー粒子と無効粒子以外
				if((particle.TYPE() != Particle::Type::Dummy) && (particle.TYPE() != Particle::Type::Disabled))
				{
					// ECS法の誤差修正項：α Dn/Dt + β (n-n0)/n0
					// α=|(n-n0)/n0|
					// β=|Dn/Dt|
					const auto speed = NeighborDensitiyVariationSpeed(i);
					const auto error = (particle.N() - n0) / n0;
					ecs[i] = std::abs(error) * speed + std::abs(speed) * error;
				}
			}
		}
#endif

		// 陽的に解く部分（第一段階）を計算する
		void ComputeExplicitForces()
		{
			const auto n0 = environment.N0();
			const auto r_e = environment.R_e;
			const auto lambda = environment.Lambda();
			const auto nu = environment.Nu;
			const auto dt = environment.Dt();
			const auto g = environment.G;

			// 加速度
			auto& a = du;

			// 全粒子で
			const auto n = particles.size();
			for(auto i = decltype(n)(0); i < n; i++)
			{
				auto& particle = particles[i];

				// 水粒子のみ
				if(particle.TYPE() == Particle::Type::IncompressibleNewton)
				{
					// 粘性の計算
					const auto vis = AccumulateNeighbor<Detail::Field::Name::U, Detail::Field::Name::X, Detail::Field::Name::Type>(i, VectorZero,
						[&thisX = particle.X(), &thisU = particle.U(), &n0, &r_e, &lambda, &nu](const Vector& u, const Vector& x, const Particle::Type type)
					{
						// ダミー粒子以外
						if(type != Particle::Type::Dummy)
						{
							const auto r = R(thisX, x);
#ifdef MPS_HL
							// HL法（高精度ラプラシアン）: ν(5-D)r_e/n0 (u_j - u_i) / r^3
							const Vector result = (nu * (5 - DIM) * r_e / n0 / (r*r*r)) * (u - thisU);
#else
							// 標準MPS法：ν*2D/λn0 (u_j - u_i) w
							const double w = Particle::W(r, r_e);
							const Vector result = (nu * 2 * DIM / lambda / n0 * w)*(u - thisU);
#endif
							return result;
						}
						else
						{
							return VectorZero;
						}
					});

					// 重力 + 粘性
					a[i] = g + vis;
				}
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

		// 陰的に解く部分（第ニ段階）を計算する
		void ComputeImplicitForces()
		{
#ifdef PRESSURE_EXPLICIT
			// 圧力を計算する
			for(unsigned int i = 0; i < particles.size(); i++)
			{
				const auto c = environment.C;
				const auto n0 = environment.N0();
				const auto rho0 = environment.Rho;

				auto& particle = particles[i];

				// 仮想的な密度：ρ0/n0 * n
				const auto rho = rho0 / n0 * particle.N();

				// 圧力の計算：c^2 (ρ-ρ0)（基準密度以下なら圧力は発生しない）
				particle.P() = (rho <= rho0) ? 0 : c*c*(rho - rho0);
			}
#else
			// 圧力方程式を設定
			SetPressurePoissonEquation();

			// 圧力方程式を解く
			SolvePressurePoissonEquation();

			// 得た圧力を代入する
			for (unsigned int i = 0; i < particles.size(); i++)
			{
				// ダミー粒子と無効粒子は除く
				if ((particles[i].TYPE() != Particle::Type::Dummy) && (particles[i].TYPE() != Particle::Type::Disabled))
				{
					const auto n0 = environment.N0();
					const auto surfaceRatio = environment.SurfaceRatio;

					// 負圧であったり自由表面の場合は圧力0
					const auto p = ppe.x(i);
					const auto n = particles[i].N();
					particles[i].P() = ((p < 0) || IsSurface(n, n0, surfaceRatio)) ? 0 : p;
				}
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
				ppe.A = Ppe::Matrix(n, n);
				ppe.x = Ppe::LongVector(n);
				ppe.b = Ppe::LongVector(n);
				ppe.cg.r = Ppe::LongVector(n);
				ppe.cg.p = Ppe::LongVector(n);
				ppe.cg.Ap = Ppe::LongVector(n);
			}
			// 全粒子で
			for (unsigned int i = 0; i < n; i++)
			{
				const auto thisN = particles[i].N();

				// ダミー粒子と無効粒子と自由表面は0
				if((particles[i].TYPE() == Particle::Type::Dummy) || (particles[i].TYPE() == Particle::Type::Disabled) || IsSurface(thisN, n0, surfaceRatio))
				{
					ppe.b(i) = 0;
					ppe.x(i) = 0;
				}
				else
				{
					// 生成項を計算する：ρ/n0 Δt Dn/Dt
					const auto speed = NeighborDensitiyVariationSpeed(i);
#ifdef MPS_ECS
					const auto e = ecs[i];
					ppe.b(i) = rho / (n0 * dt) * (speed + e);
#else
					ppe.b(i) = rho / (n0 * dt) * speed;
#endif


					// 圧力を未知数ベクトルの初期値にする
					const auto x_i = particles[i].P();
					ppe.x(i) = x_i;
				}
			}
			// TODO: 以下もそうだけど、圧力方程式を作る際にインデックス指定のfor回さなきゃいけないのが気持ち悪いので、どうにかしたい

			// 係数行列初期化
			ppe.A.clear();

			// 全粒子で
			for(auto i = decltype(n)(0); i < n; i++)
			{
				// ダミー粒子と無効粒子と自由表面は対角項だけ1
				if((particles[i].TYPE() == Particle::Type::Dummy) || (particles[i].TYPE() == Particle::Type::Disabled) || IsSurface(particles[i].N(), n0, surfaceRatio))
				{
					ppe.A(i, i) = 1.0;
				}
				else
				{
					// 対角項を設定
					ppe.A(i, i) = AccumulateNeighbor<Detail::Field::Name::ID, Detail::Field::Name::X, Detail::Field::Name::N, Detail::Field::Name::Type>(i, 0.0,
					[i, &thisX = particles[i].X(), rho, lambda, r_e, n0, surfaceRatio, &A = ppe.A](const std::size_t j, const Vector& x, const double n, const Particle::Type type)
					{
						// ダミー粒子以外
						if(type != Particle::Type::Dummy)
						{
							const auto r = R(thisX, x);
							// 非対角項を計算
#ifdef MPS_HL
							// HL法（高精度ラプラシアン）: -(5-D)r_e/n0 / r^3
							const auto a_ij = -(5 - DIM) * r_e / n0 / (r*r*r);
#else
							// 標準MPS法：-2D/(λn0) w
							const auto a_ij = (-2 * DIM / lambda * n0) * Particle::W(r, r_e);
#endif

							// 自由表面の場合は非対角項は設定しない
							if(!IsSurface(n, n0, surfaceRatio))
							{
								A(i, j) = a_ij;
							}

							return -a_ij;
						}
						else
						{
							return 0.0;
						}
					});
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

		// 圧力勾配によって速度と位置を修正する
		void ModifyByPressureGradient()
		{
			// 全粒子で
			const auto n = particles.size();
			for(auto i = decltype(n)(0); i < n; i++)
			{
				auto& particle = particles[i];

				// 水粒子のみ
				if (particle.TYPE() == Particle::Type::IncompressibleNewton)
				{
					const double r_e = environment.R_e;
					const double dt = environment.Dt();
					const double rho = environment.Rho;
					const double n0 = environment.N0();

					// 圧力勾配を計算する
#ifdef PRESSURE_GRADIENT_MIDPOINT
					// 速度修正量を計算
					const auto d = AccumulateNeighbor<Detail::Field::Name::P, Detail::Field::Name::X>(i, VectorZero,
					[&thisP = particle.P(), &thisX = particle.X(), &r_e, &dt, &rho, &n0](const double p, const Vector& x)
					{
						namespace ublas = boost::numeric::ublas;

						// 標準MPS法：-Δt/ρ D/n_0 (p_j + p_i)/r^2 w * dx
						const auto dx = x - thisX;
						const auto r2 = ublas::inner_prod(dx, dx);
						const Vector result = (-dt / rho * DIM / n0 * (p + thisP) / r2 * Particle::W(R(x, thisX), r_e)) * dx;
						return result;
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
					du[i] = d;
				}
			}

			// 全粒子で
			for (unsigned int i = 0; i < particles.size(); i++)
			{
				// 水粒子のみ
				if (particles[i].TYPE() == Particle::Type::IncompressibleNewton)
				{
					const double dt = environment.Dt();

					// 位置・速度を修正
					Vector thisDu = du[i];
					particles[i].U() += thisDu;
					particles[i].X() += thisDu * dt;
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
		Computer(
#ifndef PRESSURE_EXPLICIT
			const double allowableResidual,
#endif
			const Environment& env)
			: environment(env),
			grid(env.NeighborLength, env.L_0, env.MinX, env.MaxX),
			neighbor()
		{
#ifndef PRESSURE_EXPLICIT
			// 圧力方程式の許容誤差を設定
			ppe.allowableResidual = allowableResidual;
#endif
		}

		Computer(const Computer&) = delete;
		Computer(Computer&&) = delete;
		Computer& operator=(const Computer&) = delete;

		// 時間を進める
		void ForwardTime()
		{
			// 時間刻みを設定
			DetermineDt();

			// 近傍粒子探索
			// ※近傍粒子半径を大きめにとっているので1回で良い
			SearchNeighbor();

			// 粒子数密度を計算する
			ComputeNeighborDensities();

#ifdef MPS_ECS
			ComputeErrorCorrection();
#endif

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
		template<typename PARTICLES>
		void AddParticles(PARTICLES&& src)
		{
			particles.insert(particles.end(),
				std::make_move_iterator(src.begin()),
				std::make_move_iterator(src.end()));

			const auto n = particles.size();

			neighbor.resize(boost::extents
				[static_cast<decltype(neighbor)::index>(n)]
				[1 + static_cast<decltype(neighbor)::index>(grid.MaxParticles()*Grid::MAX_NEIGHBOR_BLOCK)]); // 先頭は近傍粒子数

			du.resize(n);
#ifdef MPS_ECS
			ecs.resize(n);
#endif
		}

		// 粒子リストを取得する
		const auto& Particles() const
		{
			return this->particles;
		}

		// 計算空間パラメーターを取得する
		const Environment& Environment() const
		{
			return environment;
		}
	};
}
#endif
