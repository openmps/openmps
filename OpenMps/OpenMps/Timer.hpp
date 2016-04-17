#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED

#include <chrono>

// 時間計測
class Timer
{
private:
	std::chrono::time_point<std::chrono::system_clock> begin;

public:
	void Start()
	{
		this->begin = std::chrono::system_clock::now();
	}

	auto Time()
	{
		const auto end = std::chrono::system_clock::now();
		return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()/1000.0;
	}
};

#endif
