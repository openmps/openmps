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

		ParticleNumberDensity(const double v)
			: val(v)
		{}

		void clear()
		{
			val = 0;
		}
	};
}

#endif
