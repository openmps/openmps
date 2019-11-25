#include <gtest/gtest.h>

#define TEST_CONJUGATEGRADIENT
#include "../Computer.hpp"

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

	// 許容する相対誤差
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
			constexpr auto Ndim = 10;
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

		// 1次元常微分方程式 d^2f/dx^2 = x を共役勾配法で解き、解析解と比較
		// f(0) = a,  f(1) = b のとき、f(x) = 1/6 x^3 + (b-a-1/6)x + a
		TEST_F(ConjugateGradientTest, Solve1dODE)
		{
			constexpr auto nx = 10; // 境界を含めた点数がnx+2
			constexpr double dx = 1.0 / (nx+1);
			constexpr double dx2 = dx * dx;

			constexpr double a = 0.0;
			constexpr double b = 1.0;

			setMatrixDim(nx);
			auto& ppe = getPpe();

			for (auto j = decltype(nx){0}; j < nx; j++)
			{
				for (auto i = decltype(nx){0}; i < nx; i++)
				{
					ppe.A(i, j) = 0.0;
				}
				ppe.b(j) = 0.0;
			}

			for (auto j = decltype(nx){1}; j < nx-1; j++)
			{
				ppe.A(j - 1, j) = 1.0/dx2;
				ppe.A(j, j) = -2.0/dx2;
				ppe.A(j + 1, j) = 1.0/dx2;

				ppe.b(j) = dx * (j+1);
			}

			ppe.A(0, 0) = -2.0/dx2;
			ppe.A(1, 0) = 1.0/dx2;
			ppe.A(nx-1, nx-1) = -2.0/dx2;
			ppe.A(nx-2, nx-1) = 1.0/dx2;

			ppe.b(0) = -1.0 / dx2 * a + dx;
			ppe.b(nx - 1) = - 1.0 / dx2 * b + (1.0-dx);

			SolvePressurePoissonEquation();

			// 解析解との相対誤差を計算
			double diff = 0.0;
			for (auto j = decltype(nx){0}; j < nx; j++)
			{
				const double x = dx * (j+1);
				const double analy = 1.0 / 6.0 * x * x * x + (b - a - 1.0 / 6.0) * x + a;
				diff += abs((ppe.x(j) - analy) / analy);
			}
			ASSERT_NEAR(diff/nx, 0.0, testAccuracy);
		}
	}

}

