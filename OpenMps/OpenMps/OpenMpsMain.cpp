#include <iostream>
#include <fstream>
#include <typeinfo>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include "Particle.hpp"
#include "MpsComputer.hpp"

// 粒子タイプを数値に変換する
inline int GetParticleTypeNum(const OpenMps::Particle& particle)
{
	return typeid(particle).hash_code();
}


// 計算結果をCSVへ出力する
void OutputToCsv(const OpenMps::MpsComputer& computer, const int& outputCount)
{
	// ファイルを開く
	auto filename = (boost::format("result/particles_%05d.csv") % outputCount).str();
	std::ofstream output(filename);

	// ヘッダ出力
	output << "Type, x, z, u, w, p, n" << std::endl;
			
	// 各粒子を出力
	for(auto particle : computer.Particles())
	{
		auto typeNum = GetParticleTypeNum(*particle);

		output
			<< typeNum << ", "
			<< particle->X() << ", " << particle->Z() << ", "
			<< particle->U() << ", " << particle->W() << ", "
			<< particle->P() << ", "
			<< particle->N() << std::endl;
	}
}

// エントリポイント
int main()
{
	system("mkdir result");
	using namespace OpenMps;

	const double l_0 = 1e-3;
	const double g = 9.8;
	const double rho = 998.20;
	const double nu = 1.004e-6;
	const double C = 0.1;
	const double r_eByl_0 = 2.9;
	const double surfaceRatio = 0.95;
	const double eps = rho*g*l_0*0.01;

	// 出力時間刻み
	const double outputInterval = 0.001;

	// 計算空間の初期化
	MpsComputer computer(outputInterval/2, g, rho, nu, C, l_0, r_eByl_0, surfaceRatio, eps);

	// 乱数生成器
	boost::minstd_rand gen(42);
	boost::uniform_real<> dst(0, l_0*0.1);
	boost::variate_generator<
	boost::minstd_rand&, boost::uniform_real<>
	> make_rand( gen, dst );

	// ダムブレーク環境を作成
	{
		const int L = 30;
		const int H = 20;

		// 水を追加
		for(int i = 0; i < L/3; i++)
		{
			for(int j = 0; j < H/1.5; j++)
			{
				double x = i*l_0;
				double y = j*l_0;

				double u = 0*make_rand()*C;
				double v = 0*make_rand()*C;

				auto particle = std::shared_ptr<Particle>(new ParticleIncompressibleNewton(x, y, u, v, 0, 0));
				computer.AddParticle(particle);
			}
		}

		// 床と天井を追加
		for(int i = -1; i < L+1; i++)
		{
			double x = i*l_0;

			// 床
			{
				// 粒子を作成して追加
				auto wall1 = std::shared_ptr<Particle>(new ParticleWall(x, -l_0*1, 0, 0));
				auto dummy1 = std::shared_ptr<Particle>(new ParticleDummy(x, -l_0*2));
				auto dummy2 = std::shared_ptr<Particle>(new ParticleDummy(x, -l_0*3));
				computer.AddParticle(wall1);
				computer.AddParticle(dummy1);
				computer.AddParticle(dummy2);
			}
			
			// 天井
			{
				
			}
		}

		// 側壁の追加
		for(int j = 0; j < H+1; j++)
		{
			double y = j*l_0;

			// 左壁
			{
				// 粒子を作成して追加
				auto wall1 = std::shared_ptr<Particle>(new ParticleWall(-l_0*1, y, 0, 0));
				auto dummy1 = std::shared_ptr<Particle>(new ParticleDummy(-l_0*2, y));
				auto dummy2 = std::shared_ptr<Particle>(new ParticleDummy(-l_0*3, y));
				computer.AddParticle(wall1);
				computer.AddParticle(dummy1);
				computer.AddParticle(dummy2);
			}
			
			// 右壁
			{
				
			}
		}

		// 四隅
		// 側壁の追加
		for(int j = 0; j < 3; j++)
		{
			double y = j*l_0;

			// 左下
			{
				// 粒子を作成して追加
				auto dummy1 = std::shared_ptr<Particle>(new ParticleDummy(-l_0*2, y-3*l_0));
				auto dummy2 = std::shared_ptr<Particle>(new ParticleDummy(-l_0*3, y-3*l_0));
				computer.AddParticle(dummy1);
				computer.AddParticle(dummy2);
			}
			
			// 左上
			{
				
			}
			
			// 右下
			{
				
			}
			
			// 右上
			{
				
			}
		}
	}

	// 初期状態を出力
	OutputToCsv(computer, 0);

	// 計算が終了するまで
	double nextOutputT = 0;
	int iteration = 0;
	for(int outputCount = 1; computer.T() < 0.2; outputCount++)
	{
		try
		{
			// 次の出力時間まで
			nextOutputT += outputInterval;
			while( computer.T() < nextOutputT)
			{
				// 時間を進める
				computer.ForwardTime();
				iteration++;
			}

			// CSVに結果を出力
			OutputToCsv(computer, outputCount);

			// 現在時刻を画面表示
			std::cout << boost::format("#%3%: t=%1% (%2%)") % computer.T() % iteration % outputCount << std::endl;
		}
		// 計算で例外があったら
		catch(MpsComputer::Exception ex)
		{
			// エラーメッセージを出して止める
			std::cout << "!!!!ERROR!!!!" << std::endl
				<< boost::format("#%3%: t=%1% (%2%)") % computer.T() % iteration % outputCount << std::endl
				<< ex.Message << std::endl;
			break;
		}
	}

	// 終了
	std::cout << "finished" << std::endl;
	system("pause");
	return 0;
}