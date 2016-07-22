#ifndef INTERACTION_INCLUDED
#define INTERACTION_INCLUDED

#pragma warning(push, 0)
#include <particle_simulator.hpp>
#pragma warning(pop)

namespace OpenMps
{
	// 粒子数密度
	struct ParticleNumberDensity
	{
		double val;

		ParticleNumberDensity()
			: val(0)
		{}

		void clear()
		{
			val = 0;
		}
	};

	// 力
	struct Force
	{
		PS::F64vec val;

		void clear()
		{
			val.x = 0;
			val.y = 0;
		}
	};
}

#endif
