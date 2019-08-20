#include <gtest/gtest.h>

#define TEST_CONSTRUCTOR
#include "../Computer.hpp"
#include "../Particle.hpp"

namespace {
#ifndef PRESSURE_EXPLICIT
	static constexpr double eps = 1e-10;
#endif

	static constexpr double dt_step = 1.0 / 100;
	static constexpr double courant = 0.1;

	static constexpr double l0 = 0.001;
	static constexpr double g = 9.8;

	static constexpr double rho = 998.2;
	static constexpr double nu = 1.004e-06;
	static constexpr double r_eByl_0 = 2.4;
	static constexpr double surfaceRatio = 0.95;
	static constexpr double minX = -0.004;
	static constexpr double minZ = -0.004;
	static constexpr double maxX = 0.053;
	static constexpr double maxZ = 0.1;

	// 格子状に配置する際の1辺あたりの粒子数
	static constexpr int num_ps_x = 7;
	static constexpr int num_ps_z = 9;

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

	class ConstructorTest : public ::testing::Test
	{
	protected:
		OpenMps::Computer<decltype(positionWall)&,decltype(positionWallPre)&> *computer;

		// それぞれのテストケースはTEST_Fが呼ばれる直前にSetUpで初期化される
		virtual void SetUp()
		{
			auto&& environment = OpenMps::Environment(dt_step, courant,
				g, rho, nu, surfaceRatio, r_eByl_0,
#ifdef PRESSURE_EXPLICIT
				c,
#endif
				l0,
				minX, minZ,
				maxX, maxZ
				);

			OpenMps::Computer<decltype(positionWall)&,decltype(positionWallPre)&> comp = OpenMps::CreateComputer(
#ifndef PRESSURE_EXPLICIT
				eps,
#endif
				environment,
				positionWall, positionWallPre);

			computer = new OpenMps::Computer<decltype(positionWall)&,decltype(positionWallPre)&>(std::move(comp));

			std::vector<OpenMps::Particle> particles;

			// 1辺l0, num_ps_x*num_ps_zの格子状に粒子を配置
			for(int j = 0; j < num_ps_z; ++j)
			{
				for(int i = 0; i < num_ps_x; ++i)
				{
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

#ifndef PRESSURE_EXPLICIT
		auto getAllowableResidual()
		{
			return computer->ppe.allowableResidual;
		}
#endif

		virtual void TearDown()
		{
			delete computer;
		}
	};

	TEST_F(ConstructorTest, FieldEnvironment)
	{
		const auto& env = computer->GetEnvironment();

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
		const double dx_courant = courant*l0;
		const double dt_grav = std::sqrt(2.0*dx_courant/g);
		ASSERT_LE( env.MaxDt, dt_step );
		ASSERT_LE( env.MaxDt, dt_grav );
#ifdef PRESSURE_EXPLICIT
		const double dt_sound = dx_courant/c;
		ASSERT_LE( env.MaxDt, dt_sound );
#endif
		// 空間刻みがCFL条件を満足するか？
		ASSERT_GE( env.MaxDx, dx_courant );

		// 近傍粒子として判定する距離が、クーラン数による距離の1倍以上か？
		ASSERT_GE( env.NeighborLength, r_eByl_0*l0*(1+courant) );
	}

	TEST_F(ConstructorTest, FieldParticles)
	{
		const auto& particles = computer->Particles();

		// 粒子数チェック
		ASSERT_EQ( particles.size(), num_ps_x*num_ps_z );

		// 辺の長さが num_ps_x*l0, num_ps_z*l0 の長方形の内部に存在するか？
		for(const auto& p : particles)
		{
			const double px = p.X()[OpenMps::AXIS_X];
			const double pz = p.X()[OpenMps::AXIS_Z];
			ASSERT_LE( px, (num_ps_x-1)*l0 );
			ASSERT_GE( px, 0 );
			ASSERT_LE( pz, (num_ps_z-1)*l0 );
			ASSERT_GE( pz, 0 );
		}
	}

	TEST_F(ConstructorTest, FieldComputer)
	{
		// 設定した定数値がフィールド値と一致するか？
#ifndef PRESSURE_EXPLICIT
		ASSERT_DOUBLE_EQ(getAllowableResidual(), eps);
#endif
	}

}
}
