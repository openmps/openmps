#pragma warning(push, 0)
#include <iostream>
#include <fstream>
#include <ctime>
#include <type_traits>
#include <boost/format.hpp>
#pragma warning(pop)

#include "Computer.hpp"
#include "Timer.hpp"

// iccはC++14の対応が遅れているので
#ifdef __INTEL_COMPILER
namespace std
{
	template<typename T>
	using underlying_type_t = typename underlying_type<T>::type;
}
#endif

// 計算結果をCSVへ出力する
static auto OutputToCsv(const OpenMps::Computer& computer, const int& outputCount)
{
	// ファイルを開く
	auto filename = (boost::format("result/particles_%05d.csv") % outputCount).str();
	std::ofstream output(filename);

	// ヘッダ出力
	output << "Type, x, z, u, w, p, n" << std::endl;

	// 各粒子を出力
	std::size_t nonDisalbeCount = 0;
	for(const auto& particle : computer.Particles())
	{
		output
			<< static_cast<std::underlying_type_t<OpenMps::Particle::Type>>(particle.TYPE()) << ", "
			<< particle.X()[0] << ", " << particle.X()[1] << ", "
			<< particle.U()[0] << ", " << particle.U()[1] << ", "
			<< particle.P() << ", "
			<< particle.N() << std::endl;

		if (particle.TYPE() != OpenMps::Particle::Type::Disabled)
		{
			nonDisalbeCount++;
		}
	}

	return nonDisalbeCount;
}

// MPS計算用の計算空間固有パラメータを作成する
static OpenMps::Environment MakeEnvironment(const double l_0, const double courant, const double outputInterval, const std::vector<OpenMps::Particle>& particles)
{
	const double g = 9.8;
	const double rho = 998.20;
	const double nu = 1.004e-6;
	const double r_eByl_0 = 2.4;
	const double surfaceRatio = 0.95;
#ifdef PRESSURE_EXPLICIT
	const double c = 1500/1000; // 物理的な音速は1500[m/s]だが、計算上小さくすることも可能
#endif
#ifdef ARTIFICIAL_COLLISION_FORCE
	const double tooNearRatio = 0.5;
	const double tooNearCoefficient = 1.5;
#endif

	// 計算空間の領域を計算
	double minX = particles.cbegin()->X()[0];
	double minZ = particles.cbegin()->X()[1];
	double maxX = minX;
	double maxZ = minZ;
	for (auto& particle : particles)
	{
		const auto x = particle.X()[0];
		const auto z = particle.X()[1];
		minX = std::min(minX, x);
		minZ = std::min(minZ, z);
		maxX = std::max(maxX, x);
		maxZ = std::max(maxZ, z);
	}

	return OpenMps::Environment(outputInterval/10, courant,
#ifdef ARTIFICIAL_COLLISION_FORCE
		tooNearRatio, tooNearCoefficient,
#endif
		g, rho, nu, surfaceRatio, r_eByl_0,
#ifdef PRESSURE_EXPLICIT
		c,
#endif
		l_0,
		minX, minZ,
		maxX, maxZ);
}

// 粒子を作成する
static auto CreateParticles(const double l_0)
{
	std::vector<OpenMps::Particle> particles;

	// ダムブレークのモデルを作成
	{
		const int L = 100;
		const int H = 200;

		// 水を追加
		for(int i = 0; i < L/2; i++)
		{
			for(int j = 0; j < H/1.5; j++)
			{
				auto particle = OpenMps::Particle(OpenMps::Particle::Type::IncompressibleNewton);

				particle.X()[0] = i*l_0;
				particle.X()[1] = j*l_0;

				particles.push_back(std::move(particle));
			}
		}

		// 床を追加
		for(int i = -1; i < L+1; i++)
		{
			const double x = i*l_0;

			// 床
			{
				auto wall1  = OpenMps::Particle(OpenMps::Particle::Type::Wall);
				auto dummy1 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy2 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy3 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				
				wall1.X()[0]  = x; wall1.X()[1]  = -l_0 * 1;
				dummy1.X()[0] = x; dummy1.X()[1] = -l_0 * 2;
				dummy2.X()[0] = x; dummy2.X()[1] = -l_0 * 3;
				dummy3.X()[0] = x; dummy3.X()[1] = -l_0 * 4;

				particles.push_back(std::move(wall1));
				particles.push_back(std::move(dummy1));
				particles.push_back(std::move(dummy2));
				particles.push_back(std::move(dummy3));
			}
		}

		// 側壁の追加
		for(int j = 0; j < H+1; j++)
		{
			const double y = j*l_0;

			// 左壁
			{
				auto wall1  = OpenMps::Particle(OpenMps::Particle::Type::Wall);
				auto dummy1 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy2 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy3 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				
				wall1.X()[0]  = -l_0 * 1; wall1.X()[1]  = y;
				dummy1.X()[0] = -l_0 * 2; dummy1.X()[1] = y;
				dummy2.X()[0] = -l_0 * 3; dummy2.X()[1] = y;
				dummy3.X()[0] = -l_0 * 4; dummy3.X()[1] = y;

				particles.push_back(std::move(wall1));
				particles.push_back(std::move(dummy1));
				particles.push_back(std::move(dummy2));
				particles.push_back(std::move(dummy3));
			}
		}

		// 副ダムの追加
		for(int j = 0; j < (H + 1)/10; j++)
		{
			const double y = j*l_0;

			// 左壁
			{
				auto wall1 = OpenMps::Particle(OpenMps::Particle::Type::Wall);
				auto dummy1 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy2 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy3 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);

				wall1.X()[0] =  l_0 * (L + 0); wall1.X()[1] =  y;
				dummy1.X()[0] = l_0 * (L + 1); dummy1.X()[1] = y;
				dummy2.X()[0] = l_0 * (L + 2); dummy2.X()[1] = y;
				dummy3.X()[0] = l_0 * (L + 3); dummy3.X()[1] = y;

				particles.push_back(std::move(wall1));
				particles.push_back(std::move(dummy1));
				particles.push_back(std::move(dummy2));
				particles.push_back(std::move(dummy3));
			}
		}

		// 四隅
		for(int j = 0; j < 4; j++)
		{
			const double y = j*l_0;

			// 左下
			{
				auto dummy1 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy2 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy3 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);

				dummy1.X()[0] = -l_0 * 2; dummy1.X()[1] = y - 4 * l_0;
				dummy2.X()[0] = -l_0 * 3; dummy2.X()[1] = y - 4 * l_0;
				dummy3.X()[0] = -l_0 * 4; dummy3.X()[1] = y - 4 * l_0;

				particles.push_back(std::move(dummy1));
				particles.push_back(std::move(dummy2));
				particles.push_back(std::move(dummy3));
			}

			// 右下
			{
				auto dummy1 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy2 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);
				auto dummy3 = OpenMps::Particle(OpenMps::Particle::Type::Dummy);

				dummy1.X()[0] = l_0 * (L + 1); dummy1.X()[1] = y - 4 * l_0;
				dummy2.X()[0] = l_0 * (L + 2); dummy2.X()[1] = y - 4 * l_0;
				dummy3.X()[0] = l_0 * (L + 3); dummy3.X()[1] = y - 4 * l_0;

				particles.push_back(std::move(dummy1));
				particles.push_back(std::move(dummy2));
				particles.push_back(std::move(dummy3));
			}
		}
	}

	// 粒子数を表示
	std::cout << particles.size() << " particles" << std::endl;

	return particles;
}
template<typename T>
static void System(T&& arg)
{
	const auto ret = system(std::forward<T>(arg));
	if(ret < 0)
	{
		throw std::runtime_error("Error!");
	}
}

// エントリポイント
int main()
{
	System("mkdir result");
	using namespace OpenMps;

	const double l_0 = 1e-3;
	const double outputInterval = 0.005;
	const double courant = 0.1;
#ifndef PRESSURE_EXPLICIT
	const double eps = 1e-10;
#endif

	// 粒子を作成
	auto particles = CreateParticles(l_0);

	// 計算空間の初期化
	Computer computer(
#ifndef PRESSURE_EXPLICIT
		eps,
#endif
		MakeEnvironment(l_0, courant, outputInterval, particles));

	// 粒子を追加
	computer.AddParticles(std::move(particles));

	// 開始時間を保存
	Timer timer;
	timer.Start();
	boost::format timeFormat("#%3$05d: t=%1$8.4lf (%2$05d), %10$12d particles, @ %4$02d/%5$02d %6$02d:%7$02d:%8$02d (%9$8.2lf)");

	{
		// 初期状態を出力
		const auto count = OutputToCsv(computer, 0);

		// 開始時間を画面表示
		const auto t = std::time(nullptr);
		const auto tm = std::localtime(&t);
		std::cout << timeFormat % 0.0 % 0 % 0
			% (tm->tm_mon + 1) % tm->tm_mday % tm->tm_hour % tm->tm_min % tm->tm_sec
			% timer.Time() % count
			<< std::endl;
	}

	// 計算が終了するまで
	double nextOutputT = 0;
	int iteration = 0;
	for(int outputCount = 1; outputCount <= 100 ; outputCount++)
	{
		double tComputer = computer.GetEnvironment().T();
		try
		{
			// 次の出力時間まで
			nextOutputT += outputInterval;
			while(tComputer < nextOutputT)
			{
				// 時間を進める
				computer.ForwardTime();
				tComputer = computer.GetEnvironment().T();
				iteration++;
			}

			// CSVに結果を出力
			const auto count = OutputToCsv(computer, outputCount);

			// 現在時刻を画面表示
			const auto t = std::time(nullptr);
			const auto tm = std::localtime(&t);
			std::cout << timeFormat % tComputer % iteration % outputCount
				% (tm->tm_mon+1) % tm->tm_mday % tm->tm_hour % tm->tm_min % tm->tm_sec
				% timer.Time() % count
				<< std::endl;
		}
		// 計算で例外があったら
		catch(Computer::Exception ex)
		{
			// エラーメッセージを出して止める
			std::cout << "!!!!ERROR!!!!" << std::endl
				<< boost::format("#%3%: t=%1% (%2%)") % tComputer % iteration % outputCount << std::endl
				<< ex.Message << std::endl;
			break;
		}
	}

	// 終了
	std::cout << "finished" << std::endl;
	return 0;
}
