#include <gtest/gtest.h>

#define TEST_PRESSUREGRADIENT
#include "../Computer.hpp"

#include <iostream>
#include <cstdio>

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
	static constexpr double minX = -10.0*l0;
	static constexpr double minZ = -10.0*l0;
	static constexpr double maxX = 10.0 * l0;
	static constexpr double maxZ = 10.0 * l0;

#ifdef PRESSURE_EXPLICIT
	static constexpr double c = 1.0;
#endif

static constexpr double testAccuracy = 1e-3;

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

	TEST_F(PressureGradientTest, GradValue)
	{
		std::vector<OpenMps::Particle> particles;
		static constexpr std::size_t num_x = 5;
		static constexpr std::size_t num_z = 5;
		constexpr auto gradp = 10.0;

		for (auto j = decltype(num_z){0}; j < num_z; j++)
		{
			for (auto i = decltype(num_x){0}; i < num_x; i++)
			{
				auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
				particle.X()[OpenMps::AXIS_X] = i * l0;
				particle.X()[OpenMps::AXIS_Z] = j * l0;

				particle.U()[OpenMps::AXIS_X] = 0.0;
				particle.U()[OpenMps::AXIS_Z] = 0.0;
				particle.P() = j*gradp;
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
		const auto prefact = (-env.Dt())/env.Rho;
		constexpr auto id = (num_x - 1) / 2 * (num_z + 1);
		const double du = p[id].U()[OpenMps::AXIS_X];
		const double dv = p[id].U()[OpenMps::AXIS_Z];
		const double dx = p[id].X()[OpenMps::AXIS_X] - (num_z-1)/2*l0;
		const double dz = p[id].X()[OpenMps::AXIS_Z] - (num_z-1)/2*l0;

		std::cout << "dv: " << dv << std::endl;
		std::cout << "gradp*prefact: " << gradp*prefact << std::endl;
		printf("dv,gradp*prefact: %f, %f", dv, gradp*prefact);

		ASSERT_NEAR(du, 0.0, testAccuracy);
		ASSERT_NEAR((dv-gradp*prefact)/(gradp*prefact), 0.0, testAccuracy);
		ASSERT_NEAR(dx, 0.0, testAccuracy);
		ASSERT_NEAR((dz-gradp*prefact*env.Dt())/(gradp*prefact*env.Dt()), 0.0, testAccuracy);
	}

}
}