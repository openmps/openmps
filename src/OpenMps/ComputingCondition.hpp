#ifndef COMPUTING_CONDITION_INCLUDED
#define COMPUTING_CONDITION_INCLUDED

#include "defines.hpp"

namespace OpenMps
{
	// MPS計算の計算条件
	class ComputingCondition final
	{
	public:

#ifndef PRESSURE_EXPLICIT
		// 収束判定誤差
		const double Eps = 1e-10;
#endif

		// 開始時刻
		const double StartTime;

		// 終了時刻
		const double EndTime;

		// 出力時間刻み
		const double OutputInterval;

#ifndef PRESSURE_EXPLICIT
		// @param eps 収束判定誤差
#endif
		// @param startTime 開始時刻
		// @param endTime 終了時刻
		// @param outputInterval 出力時間刻み
		ComputingCondition(
#ifndef PRESSURE_EXPLICIT
			const double eps,
#endif
			const double startTime,
			const double endTime,
			const double outputInterval)
			:
#ifndef PRESSURE_EXPLICIT
			Eps(eps),
#endif
			StartTime(startTime), EndTime(endTime),
			OutputInterval(outputInterval)
		{}

		ComputingCondition(ComputingCondition&&) = default;
		ComputingCondition(const ComputingCondition&) = delete;
		ComputingCondition& operator = (const ComputingCondition&) = delete;
	};
}
#endif
