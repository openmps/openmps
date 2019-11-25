#include <gtest/gtest.h>

#define TEST_CONJUGATEGRADIENT
#include "../Computer.hpp"


#include <iostream>
#include <cstdio>

namespace {
#ifndef PRESSURE_EXPLICIT
	static constexpr double eps = 1e-7;
#endif

	static constexpr double dt_step = 1.0 / 100;
	static constexpr double courant = 0.1;

	static constexpr double l0 = 0.01;
	static constexpr double g = 9.8;

	static constexpr double rho = 998.2;
	static constexpr double nu = 1.004e-06;
	static constexpr double r_eByl_0 = 1.5;

#ifndef MPS_SPP
	static constexpr double surfaceRatio = 0.95;
#endif

	static constexpr double minX = -l0;
	static constexpr double minZ = -l0;
	static constexpr double maxX = l0;
	static constexpr double maxZ = l0;

	// ãñóeÇ∑ÇÈëäëŒåÎç∑
	static constexpr double testAccuracy = 1e-3;


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

		class ConjugateGradientTest : public ::testing::Test
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

			void SolvePressurePoissonEquation()
			{
				computer->SolvePressurePoissonEquation();
			}

			auto& getPpe()
			{
				return computer->ppe;
			}

			void setMatrixDim(std::size_t n)
			{
				auto& ppe = getPpe();
				using Ppe = typename OpenMps::Computer<decltype(positionWall)&, decltype(positionWallPre)&>::Ppe;
				ppe.A = Ppe::Matrix{n,n};
				ppe.A = Ppe::Matrix{n,n};
				ppe.x = Ppe::LongVector(n);
				ppe.b = Ppe::LongVector(n);
				ppe.cg.r = Ppe::LongVector(n);
				ppe.cg.p = Ppe::LongVector(n);
				ppe.cg.Ap = Ppe::LongVector(n);
#ifdef USE_VIENNACL
				ppe.tempA = Ppe::TempMatrix(n, n);
#endif
			}

			virtual void TearDown()
			{
				delete computer;
			}
		};

		TEST_F(ConjugateGradientTest, SolveIdentityMatrix)
		{
			const auto Ndim = 10;
			setMatrixDim(Ndim);
			auto& ppe = getPpe();

			for (auto j = decltype(Ndim){0}; j < Ndim; j++)
			{
				ppe.b(j) = static_cast<double>(j);
				for (auto i = decltype(Ndim){0}; i < Ndim; i++)
				{
					if (i == j)
					{
						ppe.A(i, j) = 1.0;
					}
					else {
						ppe.A(i, j) = 0.0;
					}
				}
			}

			SolvePressurePoissonEquation();
			double diff = 0.0;
			for (auto i = decltype(Ndim){0}; i < Ndim; i++)
			{
				diff += abs(static_cast<double>(i) - ppe.x(i));
			}

			ASSERT_NEAR(diff / Ndim, 0.0, testAccuracy);
		}

		TEST_F(ConjugateGradientTest, Solve4x4Matrix)
		{
			setMatrixDim(4);
			auto& ppe = getPpe();

			ppe.A(0, 0) = 1.0;
			ppe.A(0, 1) = 1.0;
			ppe.A(0, 2) = 1.0;
			ppe.A(0, 3) = -1.0;

			ppe.A(1, 0) = 1.0;
			ppe.A(1, 1) = 1.0;
			ppe.A(1, 2) = -1.0;
			ppe.A(1, 3) = 1.0;

			ppe.A(2, 0) = 1.0;
			ppe.A(2, 1) = -1.0;
			ppe.A(2, 2) = 1.0;
			ppe.A(2, 3) = 1.0;

			ppe.A(3, 0) = -1.0;
			ppe.A(3, 1) = 1.0;
			ppe.A(3, 2) = 1.0;
			ppe.A(3, 3) = 1.0;

			ppe.b(0) = 4 * 1.0;
			ppe.b(1) = 4 * 2.0;
			ppe.b(2) = 4 * 3.0;
			ppe.b(3) = 4 * 4.0;

			SolvePressurePoissonEquation();

			ASSERT_NEAR(ppe.x(0), 2.0, testAccuracy);
			ASSERT_NEAR(ppe.x(1), 4.0, testAccuracy);
			ASSERT_NEAR(ppe.x(2), 6.0, testAccuracy);
			ASSERT_NEAR(ppe.x(3), 8.0, testAccuracy);
		}

		TEST_F(ConjugateGradientTest, Solve1DPoissonEqn)
		{
			const size_t nx = 20; // bulkÇÃé©óRìx
			double dx = 1.0 / (nx+2);
			double dx2 = dx * dx;

			double a = 0.0;
			double b = 1.0;

			setMatrixDim(nx);
			auto& ppe = getPpe();

			// É[Éçèâä˙âª
			for (auto j = decltype(nx){0}; j < nx; j++)
			{
				for (auto i = decltype(nx){0}; i < nx; i++)
				{
					ppe.A(i, j) = 0.0;
				}
				ppe.b(j) = 0.0;
			}

			// åWêîçsóÒê›íË
			for (auto j = decltype(nx){1}; j < nx-1; j++)
			{
				ppe.A(j - 1, j) = 1.0/1.00;
				ppe.A(j, j) = -2.0/1.00;
				ppe.A(j + 1, j) = 1.0/1.00;
			}

			// ã´äEèåèê›íË
			ppe.A(0, 0) = -2.0/1.00;
			ppe.A(1, 0) = 1.0/1.00;

			ppe.A(nx-1, nx-1) = -2.0/1.00;
			ppe.A(nx-2, nx-1) = 1.0/1.00;

			ppe.b(0) = -1.0 / 1.00 * a;
			ppe.b(nx - 1) = -1.0 / 1.00 * b;

			for (auto i = decltype(nx){0}; i < nx; i++)
			{
				for (auto j = decltype(nx){0}; j < nx; j++)
				{
					if (j != i)
					{
						ASSERT_NEAR(ppe.A(i, j) - ppe.A(j, i), 0.0, testAccuracy);
					}
				}
			}

			SolvePressurePoissonEquation();
				std::cout << -1 << ": " << a << std::endl;
			for (int j = 0; j < nx; ++j) {
				std::cout << j << ": " << ppe.x(j) << std::endl;
			}
				std::cout << nx+1 << ": " << b << std::endl;
		}
	}

}
