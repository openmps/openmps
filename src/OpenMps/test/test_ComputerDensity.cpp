#include <gtest/gtest.h>
#include <vector>
#include <cstdio>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "../Computer.hpp"
#include "../Particle.hpp"

namespace{
#ifndef PRESSURE_EXPLICIT
				const double eps = 1e-10;
#endif

				const double dt_step = 1.0 / 100;
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

				// 格子状に配置する際の1辺あたりの粒子数
			const int num_ps_x = 7;
			const int num_ps_z = 7;

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

			// それぞれのテストケースはTEST_Fが呼ばれる直前にSetUpで初期化される
			virtual void SetUp(){

		auto&& environment = OpenMps::Environment(dt_step, courant,
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
				eps,
#endif
			environment,
			positionWall, positionWallPre);

			computer = new OpenMps::Computer<POS_WALL,POS_WALL_PRE>(std::move(comp));

			// 粒子配置
			std::vector<OpenMps::Particle> particles;

			// 1辺l0, nx*nyの正方格子状に粒子を配置
			for(int j = 0; j < num_ps_z; ++j){
				for(int i = 0; i < num_ps_x; ++i){
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

		}

		auto getAllowableResidual(){
			return computer->ppe.allowableResidual;
		}

		virtual void TearDown(){
			delete computer;
		}
  };

	TEST_F(DensityTest, FieldEnvironment){
		auto env = computer->GetEnvironment();

		// 設定した定数値がフィールド値と一致するか？
#ifdef PRESSURE_EXPLICIT
		ASSERT_DOUBLE_EQ( env.C, c );
#endif
		// 物性値
		ASSERT_DOUBLE_EQ( env.G[OpenMps::AXIS_X], 0.0 );
		ASSERT_DOUBLE_EQ( env.G[OpenMps::AXIS_Z], -g );
		ASSERT_DOUBLE_EQ( env.Rho, rho );
		ASSERT_DOUBLE_EQ( env.Nu, nu );

		// 粒子法パラメータ
		ASSERT_DOUBLE_EQ( env.SurfaceRatio, surfaceRatio );
		ASSERT_DOUBLE_EQ( env.L_0, l0 );
		ASSERT_DOUBLE_EQ( env.R_e, r_eByl_0*l0 );

		// 計算範囲
		ASSERT_DOUBLE_EQ( env.MinX[OpenMps::AXIS_X], minX );
		ASSERT_DOUBLE_EQ( env.MinX[OpenMps::AXIS_Z], minZ );
		ASSERT_DOUBLE_EQ( env.MaxX[OpenMps::AXIS_X], maxX );
		ASSERT_DOUBLE_EQ( env.MaxX[OpenMps::AXIS_Z], maxZ );

		// 時間刻みが各特徴的スケール以下であるか？
		const double dt_grav = std::sqrt(2.0*(courant*l0)/g);
		ASSERT_LE( env.MaxDt, dt_step );
		ASSERT_LE( env.MaxDt, dt_grav );
#ifdef PRESSURE_EXPLICIT
		const double dt_sound = (courant*l0)/c;
		ASSERT_LE( env.MaxDt, dt_sound );
#endif

		// 空間刻みがCFL条件を満足するか？
		ASSERT_GE( env.MaxDx, l0*courant );

		// 近傍粒子として判定する距離が、クーラン数による距離の1倍以上か？
		ASSERT_GE( env.NeighborLength, r_eByl_0*l0 * (1+ courant) );
	}

	TEST_F(DensityTest, FieldParticles){
		auto particles = computer->Particles();

	 // 粒子数チェック
		ASSERT_EQ( particles.size(), num_ps_x*num_ps_z );

		// 辺の長さが num_ps_x*l0, num_ps_z*l0 の長方形の内部に存在するか？
		for(const auto& p : particles){
			const double px = p.X()[OpenMps::AXIS_X];
			const double pz = p.X()[OpenMps::AXIS_Z];
			ASSERT_LE( px, (num_ps_x-1)*l0 );
			ASSERT_GE( px, 0 );
			ASSERT_LE( pz, (num_ps_z-1)*l0 );
			ASSERT_GE( pz, 0 );
		}
	}

	TEST_F(DensityTest, FieldComputer){
#ifndef PRESSURE_EXPLICIT
      // 圧力方程式の許容誤差
      ASSERT_DOUBLE_EQ(getAllowableResidual(), eps);
#endif
  }

  }//openmps
}//anonymas
