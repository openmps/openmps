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
#include <iostream>

namespace{
#ifndef PRESSURE_EXPLICIT
				const double eps = 1e-10;
#endif
#ifdef ARTIFICIAL_COLLISION_FORCE
				const double tooNearRatio = 1.0;
				const double tooNearCoefficient = 1.0;
#endif

				const double startTime = 0.0;
				const double endTime = 0.5;
				const double outputInterval = 1.0;
				const std::size_t minStepCountPerOutput = 10;
				const double maxDt = outputInterval / minStepCountPerOutput;
				const double courant = 0.1;

				const double l0 = 0.001;
				const double g = 9.8;
				// 計算ステップから指定した時間スケール, dt = maxDt = 0.100
				// 重力加速度による時間スケール, dt = sqrt(2*l0/g) = 0.0143..

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

	namespace OpenMps{
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

		auto&& condition = OpenMps::ComputingCondition(
#ifndef PRESSURE_EXPLICIT
				eps,
#endif
				startTime, endTime,
				outputInterval
				);

		auto&& environment = OpenMps::Environment(maxDt, courant,
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
			delete computer;
		}
  };

	TEST_F(DensityTest, EnvConstructor){
		auto env = computer->GetEnvironment();

		// Environment TODO: PRESSURE_EXPLICITのifdef
		const double dt_grav = std::sqrt(2.0*l0/g);
		ASSERT_LE( env.MaxDt, dt_grav ); // PRESSURE_EXPLICITで分岐必要

		ASSERT_DOUBLE_EQ( env.MaxDx, l0*courant ); // PRESSURE_EXPLICITで分岐必要
		ASSERT_DOUBLE_EQ( env.L_0, l0 );
		ASSERT_DOUBLE_EQ( env.R_e, r_eByl_0*l0 );
		ASSERT_DOUBLE_EQ( env.G[0], 0.0 );
		ASSERT_DOUBLE_EQ( env.G[1], -g );
		ASSERT_DOUBLE_EQ( env.Rho, rho );
		ASSERT_DOUBLE_EQ( env.Nu, nu );
		ASSERT_DOUBLE_EQ( env.SurfaceRatio, surfaceRatio );
		ASSERT_DOUBLE_EQ( env.MinX[0], minX );
		ASSERT_DOUBLE_EQ( env.MinX[1], minZ );
		ASSERT_DOUBLE_EQ( env.MaxX[0], maxX );
		ASSERT_DOUBLE_EQ( env.MaxX[1], maxZ );
		ASSERT_GE( env.NeighborLength, r_eByl_0*l0 * (1+ courant*2) ); // 近傍粒子として保持する距離が、クーラン数による距離の二倍以上か？
	}

	}//openmps

}//anonymas

	///#ifndef PRESSURE_EXPLICIT
	///				const double eps = 1e-10;
	///					const double c = 1.0;
	/////		const double C; // env
	///#endif
	///#ifdef ARTIFICIAL_COLLISION_FORCE
	///				const double tooNearRatio = 1.0;
	///				const double tooNearCoefficient = 1.0;
	///#endif
	///
	///				const double startTime = 0.0;
	///				const double endTime = 0.5;
	///				const double outputInterval = 0.0;
	///				const std::size_t minStepCountPerOutput = 100;
