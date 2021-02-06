#ifndef TIMER_INCLUDED
#define TIMER_INCLUDED

#pragma warning(push, 0)
#pragma warning(disable : 4996)
#include <chrono>
#pragma warning(pop)
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
			return std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()/1000.0;
		}
	};
}

#endif
