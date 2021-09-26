#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED

#include <chrono>

namespace
{
	// 時間計測
	class Timer final
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
			return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000.0;
		}
	};
}

#endif
