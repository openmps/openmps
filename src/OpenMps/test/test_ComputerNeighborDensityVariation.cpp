#include <gtest/gtest.h>

#define TEST_NEIGHBORDENSITYVARIATION
#include "../Computer.hpp"

namespace {
#ifndef PRESSURE_EXPLICIT
	static constexpr double eps = 1e-10;
#endif

	static constexpr double dt_step = 1.0 / 100;
	static constexpr double courant = 0.1;

	static constexpr double l0 = 1.0;
	static constexpr double g = 0.0;

	static constexpr double rho = 998.2;
	static constexpr double nu = 1.004e-06;
	static constexpr double r_eByl_0 = 5.0;
#ifndef MPS_SPP
	static constexpr double surfaceRatio = 0.95;
#endif
	static constexpr double minX = -50.0 * l0;
	static constexpr double minZ = -50.0 * l0;
	static constexpr double maxX = 50.0 * l0;
	static constexpr double maxZ = 50.0 * l0;

#ifdef PRESSURE_EXPLICIT
	static constexpr double c = 1.0;
#endif

	static constexpr double testAccuracy = 0.10;
	static constexpr std::size_t num_x = 35;
	static constexpr std::size_t num_z = 35;
	static constexpr std::size_t wallMargin = 15;

	namespace OpenMps
	{
		double positionWall(std::size_t, double, double)
		{
			return 0.0;
		}

		double positionWallPre(double, double)
		{
			return 0.0;
		}

		class NeighborDensityVariationTest : public ::testing::Test
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

			auto& GetEnvironment()
			{
				return computer->GetEnvironment();
			}

			auto& GetParticles()
			{
				return computer->Particles();
			}

			void SearchNeighbor()
			{
				computer->SearchNeighbor();
			}

			void ComputeNeighborDensities()
			{
				computer->ComputeNeighborDensities();
			}

			virtual void TearDown()
			{
				delete computer;
			}

			auto NeighborDensityVariationSpeed(const std::size_t i)
			{
				return computer->NeighborDensityVariationSpeed(i);
			}
		};

		// 粒子は等間隔l0の格子上(num_x,num_z)に配置
		// (u,v) = (gradx x, gradz z) で発散が (gradx + gradz) である速度場を与える
		TEST_F(NeighborDensityVariationTest, ValueTestLinear)
		{
			std::vector<OpenMps::Particle> particles;

			static constexpr auto gradvx = 1.0;
			static constexpr auto gradvz = -0.3;

			auto env = GetEnvironment();

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

			// 追加した粒子に対して近傍探索処理
			SearchNeighbor();
			ComputeNeighborDensities();

			auto p = GetParticles();

			// wallMargin だけ離れた、中心付近において数値解と解析解を比較
			for (auto j = wallMargin; j < num_z - wallMargin; j++)
			{
				for (auto i = wallMargin; i < num_x - wallMargin; i++)
				{
					const auto id = i + num_x * j;
					const auto dndt = NeighborDensityVariationSpeed(id) / p[id].N();
					const auto dndt_analy = -(gradvx + gradvz);

					ASSERT_NEAR(std::abs((dndt - dndt_analy) / dndt_analy), 0.0, testAccuracy);
				}
			}
		}

		// 粒子は等間隔l0の格子上(num_x,num_z)に配置
		// (u,v) = (1/2 gradx x^2,1/3 gradz z^3) で、発散が (gradx x + gradz z^3) である速度場を与える
		TEST_F(NeighborDensityVariationTest, ValueTestPolynomial)
		{
			std::vector<OpenMps::Particle> particles;

			static constexpr auto gradvx = 1.0;
			static constexpr auto gradvz = -1.5;

			auto env = GetEnvironment();

			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i)* l0;
					const auto zij = static_cast<double>(j)* l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = 1.0 / 2.0 * gradvx * (xij * xij);
					particle.U()[OpenMps::AXIS_Z] = 1.0 / 3.0 * gradvz * (zij * zij * zij);
					particle.P() = 0.0;
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}
			computer->AddParticles(std::move(particles));

			SearchNeighbor();
			ComputeNeighborDensities();

			auto p = GetParticles();

			for (auto j = wallMargin; j < num_z - wallMargin; j++)
			{
				for (auto i = wallMargin; i < num_x - wallMargin; i++)
				{
					const auto id = i + num_x * j;
					const auto xij = static_cast<double>(i)* l0;
					const auto zij = static_cast<double>(j)* l0;

					const auto dndt = NeighborDensityVariationSpeed(id) / p[id].N();
					const auto dndt_analy = -(gradvx * xij + gradvz * zij * zij);

					ASSERT_NEAR(std::abs((dndt - dndt_analy) / dndt_analy), 0.0, testAccuracy);
				}
			}
		}
	}
}
