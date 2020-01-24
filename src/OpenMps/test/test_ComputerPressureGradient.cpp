#include <gtest/gtest.h>

#define TEST_PRESSUREGRADIENT
#include "../Computer.hpp"
#include <cmath>
#include <iostream>

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
	static constexpr double r_eByl_0 = 1.5;
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

	static constexpr double testAccuracy = 0.05;
	static constexpr std::size_t num_x = 20;
	static constexpr std::size_t num_z = 20;
	static constexpr std::size_t wallMargin = 5;

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

		class PressureGradientTest : public ::testing::Test
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

				computer = new OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>(std::move(
					OpenMps::CreateComputer(
#ifndef PRESSURE_EXPLICIT
						eps,
#endif
						environment,
						positionWall, positionWallPre)));
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

			void ModifyByPressureGradientTest()
			{
				computer->ModifyByPressureGradient();
			}

			virtual void TearDown()
			{
				delete computer;
			}
		};

		// 2次、3次の多項式からなる圧力分布のテスト
		// p(x,z) = a x^2 + b z^3 
		TEST_F(PressureGradientTest, GradValuePolynomial)
		{
			std::vector<OpenMps::Particle> particles;

			// dlは分布関数の空間変化スケール
			// 最小長さである粒子相互作用半径の数倍に設定
			// 分布関数の空間変化が激しすぎて解像度限界を超えないようにしている
			static constexpr auto dl = r_eByl_0 * 10.0;
			static constexpr auto gradpx = 1.0 / (dl*dl);
			static constexpr auto gradpz = -1.0 / (dl*dl);

			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i) * l0;
					const auto zij = static_cast<double>(j) * l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = 0.0;
					particle.U()[OpenMps::AXIS_Z] = 0.0;
					particle.P() = 1.0/2.0 * (xij*xij) * gradpx + 1.0/3.0 * (zij*zij*zij) * gradpz;
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}
			computer->AddParticles(std::move(particles));

			SearchNeighbor();
			ComputeNeighborDensities();

			ModifyByPressureGradientTest();

			auto p = GetParticles();
			auto env = GetEnvironment();
			const auto prefact = (-env.Dt()) / env.Rho;

			// 壁から離れた部分で圧力勾配を解析解と比較
			for (auto j = decltype(num_z){wallMargin}; j < num_z-wallMargin; j++)
			{
				for (auto i = decltype(num_x){wallMargin}; i < num_x-wallMargin; i++)
				{
					const auto id = i + num_x * j;
					const auto xij0 = static_cast<double>(i) * l0;
					const auto zij0 = static_cast<double>(j) * l0;
					const auto xij = p[id].X()[OpenMps::AXIS_X];
					const auto zij = p[id].X()[OpenMps::AXIS_Z];

					const double du = p[id].U()[OpenMps::AXIS_X];
					const double dv = p[id].U()[OpenMps::AXIS_Z];
					const double dx = xij - xij0;
					const double dz = zij - zij0;

					const double dpx_analy = gradpx * xij0;
					const double dpz_analy = gradpz * zij0 * zij0;
					const double dx_analy = prefact * dpx_analy * env.Dt(); 
					const double dz_analy = prefact * dpz_analy * env.Dt(); 

					ASSERT_NEAR(abs((du/prefact - dpx_analy) / dpx_analy), 0.0, testAccuracy);
					ASSERT_NEAR(abs((dv/prefact - dpz_analy) / dpz_analy), 0.0, testAccuracy);
					ASSERT_NEAR(abs((dx - dx_analy) / dx_analy), 0.0, testAccuracy);
					ASSERT_NEAR(abs((dz - dz_analy) / dz_analy), 0.0, testAccuracy);
				}
			}
		}

		// 三角関数からなる圧力分布のテスト
		// p(x,z) = a cos(kx) + b sin(kz)
		TEST_F(PressureGradientTest, GradValueTrignometric)
		{
			std::vector<OpenMps::Particle> particles;

			static constexpr auto wavek = 2.0*M_PI / (10.0*r_eByl_0);
			static constexpr auto gradpx = 1.0;
			static constexpr auto gradpz = -1.0;

			for (auto j = decltype(num_z){0}; j < num_z; j++)
			{
				for (auto i = decltype(num_x){0}; i < num_x; i++)
				{
					auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
					const auto xij = static_cast<double>(i) * l0;
					const auto zij = static_cast<double>(j) * l0;
					particle.X()[OpenMps::AXIS_X] = xij;
					particle.X()[OpenMps::AXIS_Z] = zij;

					particle.U()[OpenMps::AXIS_X] = 0.0;
					particle.U()[OpenMps::AXIS_Z] = 0.0;
					particle.P() = gradpx * cos(wavek*xij) + gradpz * sin(wavek*zij);
					particle.N() = 0.0;

					particles.push_back(std::move(particle));
				}
			}
			computer->AddParticles(std::move(particles));

			SearchNeighbor();
			ComputeNeighborDensities();

			ModifyByPressureGradientTest();

			auto p = GetParticles();
			auto env = GetEnvironment();
			const auto prefact = (-env.Dt()) / env.Rho;

			for (auto j = decltype(num_z){wallMargin}; j < num_z-wallMargin; j++)
			{
				for (auto i = decltype(num_x){wallMargin}; i < num_x-wallMargin; i++)
				{
					const auto id = i + num_x * j;
					const auto xij0 = static_cast<double>(i) * l0;
					const auto zij0 = static_cast<double>(j) * l0;
					const auto xij = p[id].X()[OpenMps::AXIS_X];
					const auto zij = p[id].X()[OpenMps::AXIS_Z];

					const double du = p[id].U()[OpenMps::AXIS_X];
					const double dv = p[id].U()[OpenMps::AXIS_Z];
					const double dx = xij - xij0;
					const double dz = zij - zij0;

					const double dpx_analy = -gradpx * sin(wavek * xij) * wavek;
					const double dpz_analy = gradpz * cos(wavek * zij) * wavek;
					const double dx_analy = prefact * dpx_analy * env.Dt(); 
					const double dz_analy = prefact * dpz_analy * env.Dt(); 

					ASSERT_NEAR(abs((du/prefact - dpx_analy) / dpx_analy), 0.0, testAccuracy);
					ASSERT_NEAR(abs((dv/prefact - dpz_analy) / dpz_analy), 0.0, testAccuracy);
					ASSERT_NEAR(abs((dx - dx_analy) / dx_analy), 0.0, testAccuracy);
					ASSERT_NEAR(abs((dz - dz_analy) / dz_analy), 0.0, testAccuracy);
				}
			}
		}


	}
}
