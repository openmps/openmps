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

// 粒子を作成する
static OpenMps::Particle::List CreateParticles(const double l_0, const double courant)
{
	system("cd");
	using namespace OpenMps;

	// 乱数生成器
	const double randFactor = 1e-10;
	boost::minstd_rand gen(42);
	boost::uniform_real<> dst(-l_0*randFactor, l_0*randFactor);
	boost::variate_generator< boost::minstd_rand&, boost::uniform_real<> > make_rand(gen, dst);
	
	// ダムブレークのモデルを作成
	Particle::List particles;
	{
		const int L = 10;
		const int H = 20;
		const int wallL = (int)(5.0*L);
		const int wallH = (int)(1.1*H);

		// 水を追加
		for(int i = 0; i < L; i++)
		{
			for(int j = 0; j < H; j++)
			{
				const double x = i*l_0 + make_rand()*courant;
				const double y = j*l_0 - make_rand()*courant;

				const double u = 0;
				const double v = 0;

				std::unique_ptr<Particle> particle(new ParticleIncompressibleNewton(x, y, u, v, 0, 0));
				particles.push_back(*particle);
			}
		}

		// 床と天井を追加
		for(int i = -1; i < wallL+1; i++)
		{
			const double x = i*l_0;

			// 床
			{
				// 粒子を作成して追加
				std::unique_ptr<Particle> wall1(new ParticleWall(x, -l_0*1, 0, 0));
				std::unique_ptr<Particle> dummy1(new ParticleDummy(x, -l_0*2));
				std::unique_ptr<Particle> dummy2(new ParticleDummy(x, -l_0*3));
				std::unique_ptr<Particle> dummy3(new ParticleDummy(x, -l_0*4));
				particles.push_back(*wall1);
				particles.push_back(*dummy1);
				particles.push_back(*dummy2);
				particles.push_back(*dummy3);
			}
		}

		// 側壁の追加
		for(int j = 0; j < wallH+1; j++)
		{
			const double y = j*l_0;

			// 左壁
			{
				// 粒子を作成して追加
				std::unique_ptr<Particle> wall1(new ParticleWall(-l_0*1, y, 0, 0));
				std::unique_ptr<Particle> dummy1(new ParticleDummy(-l_0*2, y));
				std::unique_ptr<Particle> dummy2(new ParticleDummy(-l_0*3, y));
				std::unique_ptr<Particle> dummy3(new ParticleDummy(-l_0*4, y));
				particles.push_back(*wall1);
				particles.push_back(*dummy1);
				particles.push_back(*dummy2);
				particles.push_back(*dummy3);
			}
		}

		// 四隅
		// 側壁の追加
		for(int j = 0; j < 4; j++)
		{
			const double y = j*l_0;

			// 左下
			{
				// 粒子を作成して追加
				std::unique_ptr<Particle> dummy1(new ParticleDummy(-l_0*2, y-4*l_0));
				std::unique_ptr<Particle> dummy2(new ParticleDummy(-l_0*3, y-4*l_0));
				std::unique_ptr<Particle> dummy3(new ParticleDummy(-l_0*4, y-4*l_0));
				particles.push_back(*dummy1);
				particles.push_back(*dummy2);
				particles.push_back(*dummy3);
			}
		}
	}

	// 粒子数を表示
	std::cout << particles.size() << " particles" << std::endl;

	return particles;
}
// エントリポイント
int main()
{
	system("mkdir result");
	using namespace OpenMps;

	const double l_0 = 1e-3;
	const double outputInterval = 0.001;
	const double courant = 0.1;
#ifndef PRESSURE_EXPLICIT
	const double eps = 1e-10;
#endif

	// 粒子リストと計算空間パラメーターの作成
	const Particle::List particles = CreateParticles(l_0, courant);
	const MpsEnvironment environment = MakeEnvironment(l_0, courant, outputInterval);

	// 計算空間の初期化
	MpsComputer computer(
#ifndef PRESSURE_EXPLICIT
		eps,
#endif
		environment,
		particles);

	// 初期状態を出力
	OutputToCsv(computer, 0);

	// 開始時間を保存
	boost::timer timer;
	timer.restart();
	boost::format timeFormat("#%3$05d: t=%1$8.4lf (%2$05d) @ %4$02d/%5$02d %6$02d:%7$02d:%8$02d (%9$8.2lf)");

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