#include <gtest/gtest.h>
#include <vector>
#include <cstdio>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "../Computer.hpp"
#include "../Particle.hpp"
#include "../ComputingCondition.hpp"

namespace{
	double positionWall(std::size_t, double, double){
		return 0.0;
	}

	double positionWallPre(double, double){
		return 0.0;
	}
	typedef double (&POS_WALL)(std::size_t,double,double);
	typedef double (&POS_WALL_PRE)(double,double);

	class DensityTest : public ::testing::Test
	{
		protected:
			OpenMps::Computer<POS_WALL,POS_WALL_PRE> *computer;

			// それぞれのテストケースTEST_Fが呼ばれる直前にSetUpで初期化される
			virtual void SetUp(){
				printf("Setup\n");

#ifndef PRESSURE_EXPLICIT
				const double eps = 1e-10;
#endif
#ifdef ARTIFICIAL_COLLISION_FORCE
				const double tooNearRatio = 1.0;
				const double tooNearCoefficient = 1.0;
#endif

				const double startTime = 0.0;
				const double endTime = 0.5;
				const double outputInterval = 0.0;
				const std::size_t minStepCountPerOutput = 100;
				const double courant = 0.1;
				const double l0 = 0.001;

				const double g = 9.8;
				const double rho = 998.2;
				const double nu = 1.004e-06;
				const double r_eByl_0 = 2.4;
				const double surfaceRatio = 0.95;
				const double minX = -0.004;
				const double minZ = -0.004;
				const double maxX = 0.053;
				const double maxZ = 0.1;

#ifdef PRESSURE_EXPLICIT
	const double c = 1.0;
#endif

		auto&& condition = OpenMps::ComputingCondition(
#ifndef PRESSURE_EXPLICIT
				eps,
#endif
				startTime, endTime,
				outputInterval
				);

		auto&& environment = OpenMps::Environment(outputInterval / minStepCountPerOutput, courant,
#ifdef ARTIFICIAL_COLLISION_FORCE
			tooNearRatio, tooNearCoefficient,
#endif
			g, rho, nu, surfaceRatio, r_eByl_0,
#ifdef PRESSURE_EXPLICIT
			c,
#endif
			l0,
			minX, minZ,
			maxX, maxZ
		);

		OpenMps::Computer<POS_WALL,POS_WALL_PRE> comp = OpenMps::CreateComputer(
#ifndef PRESSURE_EXPLICIT
			condition.Eps,
#endif
			environment,
			positionWall, positionWallPre);

			computer = new OpenMps::Computer<POS_WALL,POS_WALL_PRE>(std::move(comp));
			}

			virtual void TearDown(){
				printf("TearDown\n");
				delete computer;
			}

			void searchNeighbor(){
				computer->SearchNeighbor();
			}

			void computeNeighborDensities(){
				computer->ComputeNeighborDensities();
			}
  };

	TEST_F(DensityTest,calcDensity){
		std::vector<OpenMps::Particle> particles;

		// 粒子を作成して入れていく
		const double l0 = 0.1*1e-3; // 0.1 mm
//		const double re = 2.1*l0;
		const int nx = 7;
		const int ny = 7;

		// 1辺l0, nx*nyの正方格子状に粒子を配置
		for(int j = 0; j < ny; ++j){
			for(int i = 0; i < nx; ++i){
				auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
				particle.X()[OpenMps::AXIS_X] = i*l0;
				particle.X()[OpenMps::AXIS_Z] = j*l0;;

				particle.U()[OpenMps::AXIS_X] = 0.0;
				particle.U()[OpenMps::AXIS_Z] = 0.0;
				particle.P() = 0.0;
				particle.N() = 0.0;

				particles.push_back(std::move(particle));
			}
		}
		computer->AddParticles(std::move(particles));

//				computer.ForwardTime();
		searchNeighbor();
		computeNeighborDensities();

		// 24番目の粒子を狙う
		// 粒子数を表示
		std::cout << particles.size() << " particles" << std::endl;

	}

}
