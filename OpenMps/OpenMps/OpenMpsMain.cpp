#include "defines.hpp"
#include <iostream>
#include <fstream>
#include <typeinfo>
#include <ctime>
#include <omp.h>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>
#include "MpsComputer.hpp"

// 粒子タイプを数値に変換する
static int GetParticleTypeNum(const OpenMps::Particle& particle)
{
	return (int)(particle.Type());
}

// 計算結果をCSVへ出力する
static void OutputToCsv(const OpenMps::MpsComputer& computer, const int& outputCount)
{
	// ファイルを開く
	auto filename = (boost::format("result/particles_%05d.csv") % outputCount).str();
	std::ofstream output(filename);

	// ヘッダ出力
	output << "Type, x, z, u, w, p, n" << std::endl;
			
	// 各粒子を出力
	for(auto particle : computer.Particles())
	{
		auto typeNum = GetParticleTypeNum(particle);

		output
			<< typeNum << ", "
			<< particle.X() << ", " << particle.Z() << ", "
			<< particle.U() << ", " << particle.W() << ", "
			<< particle.P() << ", "
			<< particle.N() << std::endl;
	}
}

// MPS計算用の計算空間固有パラメータを作成する
static OpenMps::MpsEnvironment MakeEnvironment(const double l_0, const double courant, const double outputInterval)
{
	const double g = 9.8;
	const double rho = 998.20;
	const double nu = 1.004e-6;
	const double r_eByl_0 = 2.4;
	const double surfaceRatio = 0.95;
#ifdef PRESSURE_EXPLICIT
	const double c = 1500/1000; // 物理的な音速は1500[m/s]だが、計算上小さくすることも可能
#endif
#ifdef MODIFY_TOO_NEAR
	const double tooNearRatio = 0.5;
	const double tooNearCoefficient = 1.5;
#endif

	return OpenMps::MpsEnvironment(outputInterval/2, courant,
#ifdef MODIFY_TOO_NEAR
		tooNearRatio, tooNearCoefficient,
#endif
		g, rho, nu, surfaceRatio, r_eByl_0,
#ifdef PRESSURE_EXPLICIT
		c,
#endif
		l_0);
}

// エントリポイント
int main()
{
	system("mkdir result");
	system("cd");
	using namespace OpenMps;
	
	const double l_0 = 1e-3;
	const double outputInterval = 0.001;
	const double courant = 0.1;
#ifndef PRESSURE_EXPLICIT
	const double eps = 1e-8;
#endif

	// 計算空間パラメーターの作成
	const OpenMps::MpsEnvironment environment = MakeEnvironment(l_0, courant, outputInterval);

	// 計算空間の初期化
	MpsComputer computer(
#ifndef PRESSURE_EXPLICIT
		eps,
#endif
		environment);

	// 乱数生成器
	boost::minstd_rand gen(42);
	boost::uniform_real<> dst(0, l_0*0.1);
	boost::variate_generator< boost::minstd_rand&, boost::uniform_real<> > make_rand(gen, dst);

	// ダムブレーク環境を作成
	{
		const int L = 10;
		const int H = 20;

		// 水を追加
		for(int i = 0; i < L/2; i++)
		{
			for(int j = 0; j < H/1.5; j++)
			{
				double x = i*l_0;
				double y = j*l_0;

				double u = make_rand()*courant;
				double v = make_rand()*courant;

				auto particle = std::unique_ptr<Particle>(new ParticleIncompressibleNewton(x, y, u, v, 0, 0));
				computer.AddParticle(*particle);
			}
		}
		
		auto particle2 = std::unique_ptr<Particle>(new ParticleIncompressibleNewton(l_0*0.3, 0, 0, 0, 0, 0));
		//computer.AddParticle(particle2);

		// 床と天井を追加
		for(int i = -1; i < L+1; i++)
		{
			double x = i*l_0;

			// 床
			{
				// 粒子を作成して追加
				auto wall1 = std::unique_ptr<Particle>(new ParticleWall(x, -l_0*1, 0, 0));
				auto dummy1 = std::unique_ptr<Particle>(new ParticleDummy(x, -l_0*2));
				auto dummy2 = std::unique_ptr<Particle>(new ParticleDummy(x, -l_0*3));
				auto dummy3 = std::unique_ptr<Particle>(new ParticleDummy(x, -l_0*4));
				computer.AddParticle(*wall1);
				computer.AddParticle(*dummy1);
				computer.AddParticle(*dummy2);
				computer.AddParticle(*dummy3);
			}
		}

		// 側壁の追加
		for(int j = 0; j < H+1; j++)
		{
			double y = j*l_0;

			// 左壁
			{
				// 粒子を作成して追加
				auto wall1 = std::unique_ptr<Particle>(new ParticleWall(-l_0*1, y, 0, 0));
				auto dummy1 = std::unique_ptr<Particle>(new ParticleDummy(-l_0*2, y));
				auto dummy2 = std::unique_ptr<Particle>(new ParticleDummy(-l_0*3, y));
				auto dummy3 = std::unique_ptr<Particle>(new ParticleDummy(-l_0*4, y));
				computer.AddParticle(*wall1);
				computer.AddParticle(*dummy1);
				computer.AddParticle(*dummy2);
				computer.AddParticle(*dummy3);
			}
		}

		// 四隅
		// 側壁の追加
		for(int j = 0; j < 4; j++)
		{
			double y = j*l_0;

			// 左下
			{
				// 粒子を作成して追加
				auto dummy1 = std::unique_ptr<Particle>(new ParticleDummy(-l_0*2, y-4*l_0));
				auto dummy2 = std::unique_ptr<Particle>(new ParticleDummy(-l_0*3, y-4*l_0));
				auto dummy3 = std::unique_ptr<Particle>(new ParticleDummy(-l_0*4, y-4*l_0));
				computer.AddParticle(*dummy1);
				computer.AddParticle(*dummy2);
				computer.AddParticle(*dummy3);
			}
		}
	}

	// 粒子数を表示
	std::cout << computer.Particles().size() << " particles" << std::endl;

#ifdef _OPENMP
	#pragma omp parallel
	{
		#pragma omp master
		{
			std::cout << omp_get_num_threads() << " threads" << std::endl;
		}
	}
#endif
	
	// 初期状態を出力
	OutputToCsv(computer, 0);
	
	// 開始時間を保存
	boost::timer timer;
	timer.restart();
	auto timeFormat = boost::format("#%3$05d: t=%1$8.4lf (%2$05d) @ %4$02d/%5$02d %6$02d:%7$02d:%8$02d (%9$8.2lf)");

	// 開始時間を画面表示
	auto t = std::time(nullptr);
	auto tm = std::localtime(&t);
	std::cout << timeFormat % computer.Environment().T() % 0 % 0
				% (tm->tm_mon+1) % tm->tm_mday % tm->tm_hour % tm->tm_min % tm->tm_sec
				% timer.elapsed() << std::endl;

	// 計算が終了するまで
	double nextOutputT = 0;
	int iteration = 0;
	for(int outputCount = 1; outputCount <= 500 ; outputCount++)
	{
		double tComputer = computer.Environment().T();
		try
		{
			// 次の出力時間まで
			nextOutputT += outputInterval;
			while(tComputer < nextOutputT)
			{
				// 時間を進める
				computer.ForwardTime();
				tComputer = computer.Environment().T();
				iteration++;
			}

			// CSVに結果を出力
			OutputToCsv(computer, outputCount);

			// 現在時刻を画面表示
			auto t = std::time(nullptr);
			auto tm = std::localtime(&t);
			std::cout << timeFormat % tComputer % iteration % outputCount
				% (tm->tm_mon+1) % tm->tm_mday % tm->tm_hour % tm->tm_min % tm->tm_sec
				% timer.elapsed() << std::endl;
		}
		// 計算で例外があったら
		catch(MpsComputer::Exception ex)
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
	system("pause");
	return 0;
}