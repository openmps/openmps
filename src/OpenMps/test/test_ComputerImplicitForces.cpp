#include <gtest/gtest.h>

#define TEST_IMPLICITFORCES
#include "../Computer.hpp"

namespace {
#ifndef PRESSURE_EXPLICIT
	static constexpr double eps = 1e-10;
#endif

	static constexpr double dt_step = 1.0 / 100;
	static constexpr double courant = 0.1;

	static constexpr double l0 = 1.0;
	static constexpr double g = 9.8;

	static constexpr double rho = 998.2;
	static constexpr double nu = 1.004e-06;
	static constexpr double r_eByl_0 = 5.0;

#ifndef MPS_SPP
	static constexpr double surfaceRatio = 0.95;
#endif
	// 格子状に配置する際の1辺あたりの粒子数
	static constexpr std::size_t num_x = 15;
	static constexpr std::size_t num_z = 15;

	// サイズ上限を 横 4*l0*num_x, 縦 4*l0*num_z とする
	static constexpr double minX = -l0 * num_x * 2;
	static constexpr double minZ = -l0 * num_z * 2;
	static constexpr double maxX = l0 * num_x * 2;
	static constexpr double maxZ = l0 * num_z * 2;

	// 数値解/解析解を比較する際、壁の影響を排除するためのマージン
	static constexpr std::size_t wallMargin = 6;

	static constexpr double testAccuracy = 1e-3;

	// 数値的に微分を評価する場合に許容する相対誤差
	// 粒子数(num_x,num_z) と 近傍粒子の範囲r_eByl_0 を大きくすれば、より高精度が達成できる
	// 計算負荷の都合で 10% としている
	static constexpr double testAccuracyDerv = 1e-1;

#ifdef PRESSURE_EXPLICIT
	static constexpr double c = 1.0;
#endif

	namespace OpenMps
	{
		Vector positionWall(std::size_t, double, double)
		{

			return CreateVector(0.0, 0.0);
		}

		Vector positionWallPre(double, double)
		{
			return CreateVector(0.0, 0.0);
		}

		class ImplicitForcesTest : public ::testing::Test
		{
		protected:
			OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>* computer;

			virtual void SetUp()
			{
				auto&& environment = OpenMps::Environment(dt_step, courant,
					g, rho, nu,
#ifndef MPS_SPP
					surfaceRatio,
#endif
					r_eByl_0,
#ifdef PRESSURE_EXPLICIT
					c,
#endif
					l0,
					minX, minZ,
					maxX, maxZ
				);

				environment.Dt() = dt_step;
				environment.SetNextT();

				computer = new OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>(
#ifndef PRESSURE_EXPLICIT
					eps,
#endif
					environment,
					positionWall, positionWallPre);
			}

			auto& GetParticles()
			{
				return computer->particles;
			}

			void SearchNeighbor()
			{
				computer->SearchNeighbor();
			}

			auto& Neighbor(const std::size_t i, const std::size_t idx)
			{
				return computer->Neighbor(i, idx);
			}

			auto& NeighborCount(const std::size_t i)
			{
				return computer->NeighborCount(i);
			}

			void ComputeNeighborDensities()
			{
				computer->ComputeNeighborDensities();
			}

			void ComputeImplicitForces()
			{
				computer->ComputeImplicitForces();
			}

			void SetPressurePoissonEquation()
			{
				computer->SetPressurePoissonEquation();
			}

			void SolvePressurePoissonEquation()
			{
				computer->SolvePressurePoissonEquation();
			}

			bool IsAlive(const Particle& p, const Environment& env)
			{
				return (p.TYPE() != Particle::Type::Dummy) && (p.TYPE() != Particle::Type::Disabled)
#ifndef MPS_SPP
					&& !OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>::IsSurface(p.N(), env.N0(), surfaceRatio)
#endif
					;
			}

			auto& getPpe()
			{
				return computer->ppe;
			}

			virtual void TearDown()
			{
				delete computer;
			}

			auto NeighborDensityVariationSpeed(const std::size_t i)
			{
				return computer->NeighborDensityVariationSpeed(i);
			}

			auto& GetEnvironment()
			{
				return computer->GetEnvironment();
			}
		};

		// 係数行列は対称行列であるか？
		TEST_F(ImplicitForcesTest, MatrixSymmetry)
		{
			std::vector<OpenMps::Particle> particles;
			// 粒子を(num_x, num_z)格子上に配置
			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i)* l0;
					const auto zij = static_cast<double>(j)* l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = 0.0;
					particle.U()[OpenMps::AXIS_Z] = 0.0;
					particle.P() = 0.0;
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}
			computer->AddParticles(std::move(particles));

			SearchNeighbor();
			ComputeNeighborDensities();
			SetPressurePoissonEquation();

			auto& ppe = getPpe();
			constexpr auto Ndim = num_x * num_z;

			ASSERT_EQ(ppe.b.size(), Ndim);
			for (auto i = decltype(Ndim){0}; i < Ndim; i++)
			{
				for (auto j = decltype(Ndim){0}; j < Ndim; j++)
				{
					if (j != i)
					{
						ASSERT_NEAR(ppe.A(i, j) - ppe.A(j, i), 0.0, testAccuracy);
					}
				}
			}
		}

		// 係数行列 a_ii = -Σa_ij (i!=j) という恒等式は成立するか？
		// 境界から離れた中央粒子においてテスト
		TEST_F(ImplicitForcesTest, MatrixDiagIdentity)
		{
			std::vector<OpenMps::Particle> particles;
			// 粒子を(num_x, num_z)格子上に配置
			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i)* l0;
					const auto zij = static_cast<double>(j)* l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = 0.0;
					particle.U()[OpenMps::AXIS_Z] = 0.0;
					particle.P() = 0.0;
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}

			computer->AddParticles(std::move(particles));
			SearchNeighbor();
			ComputeNeighborDensities();
			SetPressurePoissonEquation();

			auto& ppe = getPpe();
			constexpr auto Ndim = num_x * num_z;

			constexpr auto id = (num_x - 1) / 2 * (num_z + 1); // 中央粒子のindex

			double sum_nondiag = 0.0;
			for (auto j = decltype(Ndim){0}; j < Ndim; j++)
			{
				if (j != id)
				{
					sum_nondiag += ppe.A(id, j); // disable,dummy,free surface particleは寄与しない
				}
			}
			ASSERT_NEAR(std::abs((ppe.A(id, id) - (-sum_nondiag)) / ppe.A(id, id)), 0.0, testAccuracy);
		}

		// 粒子i の 近傍粒子j に対応する成分が a_ij != 0 であること
		// (dummy, disable, free surface粒子は除外)
		TEST_F(ImplicitForcesTest, MatrixNeighborNonzero)
		{
			std::vector<OpenMps::Particle> particles;
			// 粒子を(num_x, num_z)格子上に配置
			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i)* l0;
					const auto zij = static_cast<double>(j)* l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = 0.0;
					particle.U()[OpenMps::AXIS_Z] = 0.0;
					particle.P() = 0.0;
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}
			computer->AddParticles(std::move(particles));

			SearchNeighbor();
			ComputeNeighborDensities();
			SetPressurePoissonEquation();

			auto env = GetEnvironment();
			auto& ppe = getPpe();
			constexpr auto Ndim = num_x * num_z;

			for (auto i = decltype(Ndim){0}; i < Ndim; i++)
			{
				if (!IsAlive(particles[i], env))
				{
					continue;
				}

				std::vector<decltype(i)> neighList; // iの近傍粒子リスト
				for (auto idx = decltype(i){0}; idx < NeighborCount(i); idx++)
				{
					const auto j = Neighbor(i, idx);
					if (IsAlive(particles[j], env))
					{
						neighList.push_back(j);
					}
				}

				for (auto j = decltype(Ndim){0}; j < Ndim; j++)
				{
					if (j != i && std::find(neighList.begin(), neighList.end(), j) == neighList.end()) // jがi近傍リストに属しない
					{
						ASSERT_NEAR(ppe.A(i, j), 0.0, testAccuracy);
					}
				}

			}
		}

		// 行列成分値の解析値との一致をテスト
		TEST_F(ImplicitForcesTest, MatrixValueTest)
		{
			std::vector<OpenMps::Particle> particles;

			static constexpr auto gradvx = 1.0;
			static constexpr auto gradvz = -0.3;

			// 粒子を(num_x, num_z)格子上に配置
			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i)* l0;
					const auto zij = static_cast<double>(j)* l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = gradvx * xij;
					particle.U()[OpenMps::AXIS_Z] = gradvz * zij;
					particle.P() = 0.0;
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}
			computer->AddParticles(std::move(particles));

			SearchNeighbor();
			ComputeNeighborDensities();

			SetPressurePoissonEquation();

			auto& ppe = getPpe();
			constexpr auto Ndim = num_x * num_z;

			auto env = GetEnvironment();
			auto p = GetParticles();

			for (auto iz = decltype(num_z){wallMargin}; iz < num_z - wallMargin; iz++)
			{
				for (auto ix = decltype(num_x){wallMargin}; ix < num_x - wallMargin; ix++)
				{
					const auto id = ix + num_x * iz;

					if (IsAlive(p[id], env))
					{
						const auto drhodt_analy = -(gradvx + gradvz) * p[id].N();
						const auto b_analy = -env.Rho / (env.Dt() * env.N0()) * drhodt_analy;
						ASSERT_NEAR(std::abs((ppe.b(id) - b_analy) / b_analy), 0.0, testAccuracyDerv);
					}
				}
			}

			// 粒子i,jの格子座標(ix,iz), (jx,jz)
			for (auto jz = decltype(num_z){wallMargin}; jz < num_z - wallMargin; jz++)
			{
				for (auto jx = decltype(num_x){wallMargin}; jx < num_x - wallMargin; jx++)
				{
					for (auto iz = decltype(num_z){wallMargin}; iz < num_z - wallMargin; iz++)
					{
						for (auto ix = decltype(num_x){wallMargin}; ix < num_x - wallMargin; ix++)
						{
							const auto id_i = ix + num_x * iz;
							const auto id_j = jx + num_x * jz;

							// 対角成分の値/行列の対称性は別テストに任せるため、本テストでは行列の半分のみ調べる
							// + 計算に含めない粒子 と 隣接していない粒子 に対応する成分を除外
							if (id_i >= id_j ||
								(!IsAlive(p[id_i], env) || !IsAlive(p[id_j], env)) ||
								ppe.A(id_i, id_j) == 0.0)
							{
								continue;
							}

							if (IsAlive(p[id_i], env) && IsAlive(p[id_j], env))
							{
								const auto dxij = (static_cast<double>(ix) - static_cast<double>(jx))* l0;
								const auto dzij = (static_cast<double>(iz) - static_cast<double>(jz))* l0;
								const auto Rij = std::sqrt((dxij) * (dxij)+(dzij) * (dzij));

								const auto Aij_analy = (5 - DIM) * env.R_e / env.N0() / (Rij * Rij * Rij);
								ASSERT_NEAR(std::abs((ppe.A(id_i, id_j) - Aij_analy) / ppe.A(id_i, id_j)), 0.0, testAccuracyDerv);
							}

						}
					}
				}
			}

		}

	}

}

