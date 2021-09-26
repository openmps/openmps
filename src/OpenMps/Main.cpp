﻿#include <iostream>
#include <fstream>
#include <ctime>
#include <type_traits>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "ComputingCondition.hpp"
#include "Computer.hpp"
#include "Timer.hpp"
#include "stov.hpp"

// iccはC++14の対応が遅れているので
#ifdef __INTEL_COMPILER
namespace { namespace std
{
	template<typename T>
	using underlying_type_t = typename underlying_type<T>::type;
}}
#endif

namespace
{
	// 計算結果をCSVへ出力する
	template<typename COMPUTER>
	inline auto OutputToCsv(const COMPUTER& computer, const std::size_t& outputCount)
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
	inline auto InputFromCsv(const std::string& csv)
	{
		std::vector<OpenMps::Particle> particles;

		std::vector<std::string> input;
		boost::algorithm::split(input, csv, boost::is_any_of("\n"));

		// 先頭の空行は飛ばす
		auto itr = input.cbegin();
		for (; itr->size() == 0; ++itr) {}

		// 行をカンマ区切りで分割する関数
		const auto GetItems = [](std::string str)
		{
			str.erase(std::remove_if(str.begin(), str.end(),
				[](const char c)
				{
					return (c == ' ') || (c == '\t');
				}), str.end());

			std::vector<std::string> data;
			if (str.size() > 0)
			{
				boost::algorithm::split(data, str, boost::is_any_of(","));
			}
			return data;
		};

		// ヘッダー項目の列番号を取得（ない場合は0が入る）
		constexpr std::size_t HEADER_NOT_FOUND = 0;
		auto header = std::unordered_map<std::string, std::size_t>(
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
			auto& line = *itr; ++itr;
			const auto headerItems = GetItems(line);
			for (auto i = decltype(headerItems.size())(0); i < headerItems.size(); i++)
			{
				const auto name = headerItems[i];
				if (header.find(name) != header.end())
				{
					header[name] = static_cast<decltype(header)::mapped_type>(i + 1); // 列番号+1を格納しておく（0が「ない」を表すため）
				}
				else
				{
					throw std::runtime_error("Illegal header item in input csv");
				}
			}

			for (auto& item : header)
			{
				// ない項目があったらエラー
				if (item.second == HEADER_NOT_FOUND)
				{
					throw std::runtime_error("Some header item doesn't exist");
				}

				// 列番号に戻す
				item.second--;
			}
		}

		// 粒子を作成して入れていく
		for (;itr != input.cend(); ++itr)
		{
			auto& line = *itr;
			const auto data = GetItems(line);
			if (data.size() > 0) // 空行は飛ばす
			{
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
		}

		// 粒子数を表示
		std::cout << particles.size() << " particles" << std::endl;

		return particles;
	}

	// 粒子の初期状態を読み込む
	inline decltype(auto) LoadParticles(const boost::property_tree::ptree& xml)
	{
		// 粒子データの読み込み
		const auto type = xml.get_optional<std::string>("openmps.particles.<xmlattr>.type").get();
		if (type == "csv")
		{
			auto&& txt = xml.get<std::string>("openmps.particles");
			return InputFromCsv(txt);
		}
		else
		{
			throw std::runtime_error("Not Implemented!");
		}
	}

	// 計算環境を読み込む
	inline decltype(auto) LoadEnvironment(const boost::property_tree::ptree& xml, const double outputInterval)
	{
		const auto l_0 = xml.get<double>("openmps.environment.l_0.<xmlattr>.value");
		const auto minStepCountPerOutput = xml.get<std::size_t>("openmps.environment.minStepCountPerOutput.<xmlattr>.value");
		const double courant = xml.get<double>("openmps.environment.courant.<xmlattr>.value");

		const double g = xml.get<double>("openmps.environment.g.<xmlattr>.value");
		const double rho = xml.get<double>("openmps.environment.rho.<xmlattr>.value");
		const double nu = xml.get<double>("openmps.environment.nu.<xmlattr>.value");
		const double r_eByl_0 = xml.get<double>("openmps.environment.r_eByl_0.<xmlattr>.value");
#ifndef MPS_SPP
		const double surfaceRatio = xml.get<double>("openmps.environment.surfaceRatio.<xmlattr>.value");
#endif
#ifdef PRESSURE_EXPLICIT
		const double c = xml.get<double>("openmps.environment.c.<xmlattr>.value");
#endif
#ifdef ARTIFICIAL_COLLISION_FORCE
		const double tooNearRatio = xml.get<double>("openmps.environment.tooNearRatio.<xmlattr>.value");
		const double tooNearCoefficient = xml.get<double>("openmps.environment.tooNearCoefficient.<xmlattr>.value");
#endif

		const double minX = xml.get<double>("openmps.environment.minX.<xmlattr>.value");
#ifdef DIM3
		const double minY = xml.get<double>("openmps.environment.minY.<xmlattr>.value");
#endif
		const double minZ = xml.get<double>("openmps.environment.minZ.<xmlattr>.value");
		const double maxX = xml.get<double>("openmps.environment.maxX.<xmlattr>.value");
#ifdef DIM3
		const double maxY = xml.get<double>("openmps.environment.maxY.<xmlattr>.value");
#endif
		const double maxZ = xml.get<double>("openmps.environment.maxZ.<xmlattr>.value");

		return OpenMps::Environment(outputInterval / static_cast<double>(minStepCountPerOutput), courant,
#ifdef ARTIFICIAL_COLLISION_FORCE
			tooNearRatio, tooNearCoefficient,
#endif
			g, rho, nu,
#ifndef MPS_SPP
			surfaceRatio,
#endif
			r_eByl_0,
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

	// 計算条件を読み込む
	inline decltype(auto) LoadCondition(const boost::property_tree::ptree& xml)
	{
		const auto startTime = xml.get<double>("openmps.condition.startTime.<xmlattr>.value");
		const auto endTime = xml.get<double>("openmps.condition.endTime.<xmlattr>.value");
		const auto outputInterval = xml.get<double>("openmps.condition.outputInterval.<xmlattr>.value");
#ifndef PRESSURE_EXPLICIT
		const auto eps = xml.get<double>("openmps.condition.eps.<xmlattr>.value");
#endif

		return OpenMps::ComputingCondition(
#ifndef PRESSURE_EXPLICIT
			eps,
#endif
			startTime, endTime,
			outputInterval
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
}

// エントリポイント
int main(const int argc, const char* const argv[])
{
	System("mkdir result");

	const auto filename = (argc == 1) ? "../../Benchmark/Sample/Sample.xml" : argv[1];
	std::cout << "Input XML file: " << filename << std::endl;
	auto xml = std::make_unique<boost::property_tree::ptree>();
	boost::property_tree::read_xml(filename, *xml);

	auto&& condition = LoadCondition(*xml);
	auto&& environment = LoadEnvironment(*xml, condition.OutputInterval);
	auto&& particles = LoadParticles(*xml);

	xml.reset(nullptr); // 読み込んだテキストデータを強制的に廃棄

	// 粒子の初期位置を保存
	auto initialPosition = std::make_unique<OpenMps::Vector[]>(particles.size());
	std::transform(particles.cbegin(), particles.cend(), initialPosition.get(),
		[](auto particle)
		{
			return particle.X();
		});

	// 壁は動かさない
	const auto positionWall = [&initialPosition](auto i, auto, auto)
	{
		return initialPosition[i];
	};
	const auto positionWallPre = [](auto, auto)
	{
	};

	// 計算空間の初期化
	auto computer = OpenMps::CreateComputer(
#ifndef PRESSURE_EXPLICIT
		condition.Eps,
#endif
		environment,
		positionWall, positionWallPre);

	// 粒子を追加
	computer.AddParticles(std::move(particles));

	// 開始時間を保存
	Timer timer;
	timer.Start();
	boost::format timeFormat("#%3$05d: t=%1$8.4lf (%2$05d), %10$12d particles, @ %4$02d/%5$02d %6$02d:%7$02d:%8$02d (%9$8.2lf)");

	const auto outputIterationOffset = static_cast<std::size_t>(std::ceil(condition.StartTime / condition.OutputInterval));

	// 開始時間を画面表示
	const auto printTime = [&timeFormat, &timer](const auto tComputer, const auto iteration, const auto outputCount, const auto count)
	{
		const auto t = std::time(nullptr);
#ifdef _MSC_VER
		auto time = tm{};
		const auto e = ::localtime_s(&time, &t);
#else
		const auto time = *std::localtime(&t);
#endif
		std::cout << timeFormat % tComputer % iteration % outputCount
			% (time.tm_mon + 1) % time.tm_mday % time.tm_hour % time.tm_min % time.tm_sec
			% timer.Time() % count
			<< std::endl;
	};
	{
		// 初期状態を出力
		const auto count = OutputToCsv(computer, outputIterationOffset);
		printTime(condition.StartTime, 0, outputIterationOffset, count);
	}

	// 計算が終了するまで
	double nextOutputT = 0;
	std::size_t iteration = 0;
	const auto endCount = static_cast<std::size_t>(std::ceil((condition.EndTime - condition.StartTime) / condition.OutputInterval));
	for(auto outputCount = decltype(endCount){1}; outputCount <= endCount; outputCount++)
	{
		double tComputer = computer.GetEnvironment().T();
		try
		{
			// 次の出力時間まで
			nextOutputT += condition.OutputInterval;
			while(tComputer < nextOutputT)
			{
				// 時間を進める
				computer.ForwardTime();
				tComputer = computer.GetEnvironment().T();
				iteration++;
			}

			// CSVに結果を出力
			const auto count = OutputToCsv(computer, outputCount + outputIterationOffset);

			tComputer += condition.StartTime;

			// 現在時刻を画面表示
			printTime(tComputer, iteration, outputCount + outputIterationOffset, count);
		}
		// 計算で例外があったら
		catch(decltype(computer)::Exception ex)
		{
			// エラーメッセージを出して止める
			std::cout << "!!!!ERROR!!!!" << std::endl
				<< boost::format("#%3%: t=%1% (%2%)") % tComputer % iteration % (outputCount + outputIterationOffset) << std::endl
				<< ex.what() << std::endl;
			break;
		}
	}

	// 終了
	std::cout << "finished" << std::endl;
	return 0;
}
