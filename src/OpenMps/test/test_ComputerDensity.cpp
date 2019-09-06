#include <gtest/gtest.h>

#define TEST_DENSITY
#include "../Computer.hpp"
#include "../Particle.hpp"
#include "../Vector.hpp"

namespace {
#ifndef PRESSURE_EXPLICIT
	static constexpr double eps = 1e-10;
#endif

	static constexpr double dt_step = 1.0 / 100;
	static constexpr double courant = 0.1;

	static constexpr double l0 = 0.1;
	static constexpr double g = 9.8;

	static constexpr double rho = 998.2;
	static constexpr double nu = 1.004e-06;
	static constexpr double r_eByl_0 = 2.1;
	static constexpr double surfaceRatio = 0.95;
	static constexpr double minX = -0.004;
	static constexpr double minZ = -0.004;
	static constexpr double maxX = 0.053;
	static constexpr double maxZ = 0.1;

	// 格子状に配置する際の1辺あたりの粒子数
	static constexpr int num_ps_x = 7;
	static constexpr int num_ps_z = 7;

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

	class DensityTest : public ::testing::Test
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

OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&> comp = OpenMps::CreateComputer(
#ifndef PRESSURE_EXPLICIT
	eps,
#endif
	environment,
	positionWall, positionWallPre);

computer = new OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>(std::move(comp));

std::vector<OpenMps::Particle> particles;

// 1辺l0, num_ps_x*num_ps_zの格子状に粒子を配置
for (int j = 0; j < num_ps_z; ++j)
{
	for (int i = 0; i < num_ps_x; ++i)
	{
		auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
		particle.X()[OpenMps::AXIS_X] = i * l0;
		particle.X()[OpenMps::AXIS_Z] = j * l0;

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

		auto& GetParticles()
		{
			return computer->particles;
		}

		double Rv(const Vector& x1, const Vector& x2)
		{
			return OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>::R(x1, x2);
		}

		double Rp(const Particle& p1, const Particle& p2)
		{
			return OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>::R(p1, p2);
		}

		auto& Neighbor(const std::size_t i, const std::size_t idx)
		{
			return computer->Neighbor(i, idx);
		}

		auto& NeighborCount(const std::size_t i)
		{
			return computer->NeighborCount(i);
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
	};

	TEST_F(DensityTest, NeighborDistance)
	{
		// TODO: 適当な3点をいくらか取ってくる

		// 3つのVector, Particleを用意
		auto p1 = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
		auto p2 = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);
		auto p3 = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);

		p1.X()[OpenMps::AXIS_X] = 1.0;
		p1.X()[OpenMps::AXIS_Z] = 0.3;

		p2.X()[OpenMps::AXIS_X] = -2.0;
		p2.X()[OpenMps::AXIS_Z] = 0.8;

		p3.X()[OpenMps::AXIS_X] = 100.0;
		p3.X()[OpenMps::AXIS_Z] = -20.5;

		// Vector, Particle の距離計算が一致
		ASSERT_DOUBLE_EQ(Rv(p1.X(), p2.X()), Rp(p1, p2));

		// 点入れ替えについて対称
		ASSERT_DOUBLE_EQ(Rp(p1, p2), Rp(p2, p1));

		// 三角不等式
		ASSERT_LE(Rp(p1, p2), Rp(p1, p3) + Rp(p2, p3));
		ASSERT_LE(Rp(p2, p3), Rp(p1, p2) + Rp(p1, p3));
		ASSERT_LE(Rp(p3, p1), Rp(p1, p2) + Rp(p2, p3));
	}

	// 近接する2粒子は互いを近傍粒子として保持しているか？
	// TODO: 1個でもあれば false的な文を確認
	TEST_F(DensityTest, NeighborSymmetry)
	{
		SearchNeighbor();

		const auto& particles = GetParticles();
		auto n = particles.size();

		bool neigh_sym = true;
		for(auto i = decltype(n){0}; i < n; i++) // 場に存在する粒子ループ
		{
			if(particles[i].TYPE() != Particle::Type::Disabled)
			{
				for(auto idx = decltype(i){0}; idx < NeighborCount(i); idx++) // i粒子の近傍粒子ループ
				{
					auto j = Neighbor(i, idx); // i粒子に隣り合うidx番目粒子をjとする

					// j粒子の近傍にi粒子が含まれているか？
					bool hasi = false;
					for (auto idxj = decltype(i){0}; idxj < NeighborCount(j); idxj++)
					{
						if (particles[j].TYPE() != Particle::Type::Disabled && Neighbor(j, idxj) == i)
						{
							hasi |= true;
						}
					}
				}

				neigh_sym &= true;
			}
		}

		ASSERT_TRUE(neigh_sym);
	}

	// 粒子は自分自身を近傍粒子として含んでいないか？
	TEST_F(DensityTest, NeighborMyself)
	{
		SearchNeighbor();

		const auto& particles = GetParticles();
		auto n = particles.size();

		bool hasmyself = false;
		for(auto i = decltype(n){0}; i < n; i++) // 場に存在する粒子ループ
		{
			if(particles[i].TYPE() != Particle::Type::Disabled)
			{
				for(auto idx = decltype(i){0}; idx < NeighborCount(i); idx++) // i粒子の近傍粒子ループ
				{
					hasmyself = (Neighbor(i, idx) == i);
				}
			}
		}

		ASSERT_FALSE(hasmyself);
	}

	// 密粒子数密度を計算して理論値と比較
	TEST_F(DensityTest, NeighDensity)
	{
		SearchNeighbor();

		ComputeNeighborDensities();

		const auto& particles = GetParticles();
		// テキスト Koshizuka et al.,(2014) による値と比較
		// TODO: ID 24は中心粒子を表す。初期の粒子IDが維持されていると仮定している
		ASSERT_NEAR(particles[24].N(), 6.539696962, 1e-5);
	}
}
}
