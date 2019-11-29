#include <gtest/gtest.h>

#define TEST_PRESSUREGRADIENT
#include "../Computer.hpp"

#include <iostream>
#include <cmath>

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
	static constexpr double minX = -10*l0;
	static constexpr double minZ = -10*l0;
	static constexpr double maxX = 10 * l0;
	static constexpr double maxZ = 10 * l0;

	// 格子状に配置する際の1辺あたりの粒子数
	static constexpr std::size_t num_x = 5;
	static constexpr std::size_t num_z = 5;

#ifdef PRESSURE_EXPLICIT
	static constexpr double c = 1.0;
#endif

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
		OpenMps::Computer<decltype(positionWall)&,decltype(positionWallPre)&> *computer;

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

	TEST_F(PressureGradientTest, DistanceValue)
	{
		std::vector<OpenMps::Particle> particles;

		// 1辺l0, num_x*num_zの格子状に粒子を配置
		for (auto j = decltype(num_z){0}; j < num_z; j++)
		{
			for (auto i = decltype(num_x){0}; i < num_x; i++)
			{
				auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
				particle.X()[OpenMps::AXIS_X] = i * l0;
				particle.X()[OpenMps::AXIS_Z] = j * l0;

				particle.U()[OpenMps::AXIS_X] = 0.0;
				particle.U()[OpenMps::AXIS_Z] = 0.0;
				particle.P() = j*10.0;
				particle.N() = 0.0;

				particles.push_back(std::move(particle));
			}
		}
		computer->AddParticles(std::move(particles));

		SearchNeighbor();
		ComputeNeighborDensities();

		ModifyByPressureGradientTest();

		// 中心idxがほしい
		auto p = GetParticles();
		auto env = GetEnvironment();
		const double r_e = env.R_e;
		const double dt = env.Dt();
		const double rho = env.Rho;
		const double n0 = env.N0();

		double vx = p[12].U()[OpenMps::AXIS_X];
		double vz = p[12].U()[OpenMps::AXIS_Z];
		auto prefact = (-dt / rho * DIM / n0);
		std::cout << "dpx: " << vx/prefact << std::endl;
		std::cout << "dpz: " << vz/prefact << std::endl;
	}

}
}
