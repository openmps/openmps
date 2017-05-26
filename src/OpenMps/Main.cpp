#pragma warning(push, 0)
#include <iostream>
#include <fstream>
#include <ctime>
#include <type_traits>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <boost/format.hpp>
#pragma warning(pop)

#include "Computer.hpp"
#include "Timer.hpp"
#include "stov.hpp"

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
#ifdef DIM3
	output << "Type, x, y, z, u, v, w, p, n" << std::endl;
#else
	output << "Type, x, z, u, w, p, n" << std::endl;
#endif

	// 各粒子を出力
	std::size_t nonDisalbeCount = 0;
	for(const auto& particle : computer.Particles())
	{
		output
			<< static_cast<std::underlying_type_t<OpenMps::Particle::Type>>(particle.TYPE()) << ", "
#ifdef DIM3
			<< particle.X()[0] << ", " << particle.X()[1] << ", " << particle.X()[2] << ", "
			<< particle.U()[0] << ", " << particle.U()[1] << ", " << particle.U()[2] << ", "
#else
			<< particle.X()[0] << ", " << particle.X()[1] << ", "
			<< particle.U()[0] << ", " << particle.U()[1] << ", "
#endif
			<< particle.P() << ", "
			<< particle.N() << std::endl;

		if (particle.TYPE() != OpenMps::Particle::Type::Disabled)
		{
			nonDisalbeCount++;
		}
	}

	return nonDisalbeCount;
}

// 粒子を読み込む
static auto InputFromCsv()
{
	std::vector<OpenMps::Particle> particles;

	// ファイルを開く
	auto filename = "particles.csv";
	std::ifstream input(filename);

	if (input.fail())
	{
		throw std::runtime_error("Input csv doesn't exist!");
	}

	// 行をカンマ区切りで分割する関数
	const auto GetItems = [](std::string str)
	{
		str.erase(std::remove_if(str.begin(), str.end(),
			[](const char c)
		{
			return (c == ' ') || (c == '\t');
		}), str.end());

		std::string item = "";
		std::istringstream lineStream(str);
		std::vector<std::string> data;
		while (std::getline(lineStream, item, ','))
		{
			data.push_back(item);
		}
		return data;
	};


	// ヘッダー項目の列番号を取得（ない場合は-1が入る）
	constexpr std::int8_t HEADER_NOT_FOUND = -1;
	auto header = std::unordered_map<std::string, std::int8_t>(
	{
		{ "Type", HEADER_NOT_FOUND },
		{ "x", HEADER_NOT_FOUND },
#ifdef DIM3
		{ "y", HEADER_NOT_FOUND },
#endif
		{ "z", HEADER_NOT_FOUND },
		{ "u", HEADER_NOT_FOUND },
#ifdef DIM3
		{ "v", HEADER_NOT_FOUND },
#endif
		{ "w", HEADER_NOT_FOUND },
		{ "p", HEADER_NOT_FOUND },
		{ "n", HEADER_NOT_FOUND },
	});
	{
		// 先頭行を読み込み
		std::string line;
		if (!std::getline(input, line))
		{
			throw std::runtime_error("Input csv has no header");
		}
		const auto headerItems = GetItems(line);
		for (auto i = decltype(headerItems.size())(0); i < headerItems.size(); i++)
		{
			const auto name = headerItems[i];
			if (header.find(name) != header.end())
			{
				header[name] = i;
			}
			else
			{
				throw std::runtime_error("Illegal header item in input csv");
			}
		}

		// ない項目があったらエラー
		for (const auto& item : header)
		{
			if (item.second == HEADER_NOT_FOUND)
			{
				throw std::runtime_error("Some header item doesn't exist");
			}
		}
	}

	// 粒子を作成して入れていく
	std::string line;
	while (std::getline(input, line))
	{
		const auto data = GetItems(line);

		auto particle = OpenMps::Particle(static_cast<OpenMps::Particle::Type>(stov<std::underlying_type_t<OpenMps::Particle::Type>>(data[header["Type"]])));
		
		// 位置ベクトル
		particle.X()[OpenMps::AXIS_X] = stov<double>(data[header["x"]]);
#ifdef DIM3
		particle.X()[OpenMps::AXIS_Y] = stov<double>(data[header["y"]]);
#endif
		particle.X()[OpenMps::AXIS_Z] = stov<double>(data[header["z"]]);

		// 速度ベクトル
		particle.U()[OpenMps::AXIS_X] = stov<double>(data[header["u"]]);
#ifdef DIM3
		particle.U()[OpenMps::AXIS_Y] = stov<double>(data[header["v"]]);
#endif
		particle.U()[OpenMps::AXIS_Z] = stov<double>(data[header["w"]]);

		// 圧力
		particle.P() = stov<double>(data[header["p"]]);

		// 粒子数密度
		particle.N() = stov<double>(data[header["n"]]);

		particles.push_back(std::move(particle));
	}


	// 粒子数を表示
	std::cout << particles.size() << " particles" << std::endl;

	return particles;
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
#ifdef DIM3
	double minX = particles.cbegin()->X()[0];
	double minY = particles.cbegin()->X()[1];
	double minZ = particles.cbegin()->X()[2];
	double maxX = minX;
	double maxY = minY;
	double maxZ = minZ;
	for (auto& particle : particles)
	{
		const auto x = particle.X()[0];
		const auto y = particle.X()[1];
		const auto z = particle.X()[2];
		minX = std::min(minX, x);
		minY = std::min(minY, y);
		minZ = std::min(minZ, z);
		maxX = std::max(maxX, x);
		maxY = std::max(maxY, y);
		maxZ = std::max(maxZ, z);
	}
#else
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
#endif

	return OpenMps::Environment(outputInterval/10, courant,
#ifdef ARTIFICIAL_COLLISION_FORCE
		tooNearRatio, tooNearCoefficient,
#endif
		g, rho, nu, surfaceRatio, r_eByl_0,
#ifdef PRESSURE_EXPLICIT
		c,
#endif
		l_0,
#ifdef DIM3
		minX, minY, minZ,
		maxX, maxY, maxZ
#else
		minX, minZ,
		maxX, maxZ
#endif
	);
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
	auto particles = InputFromCsv();

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
