# OpenMps

This project contains an implemention of Moving Particle Semi-implicit (MPS) method.
MPS method is one of the most popular Particle Method to solve Continuum Dynamics (e.g. Fluid Dynamics). See [Wikipedia](https://en.wikipedia.org/wiki/Moving_Particle_Semi-implicit_Method) for more information.

![screenshot](https://bytebucket.org/OpenMps/openmps.bitbucket.org/raw/master/img/icon.PNG)

## License

All files in this repository, including which has no header text about the license, are licensed under [GNU General Public License (GPL) v3.0 or later](http://www.gnu.org/licenses/gpl.html) (See LICENSE).

```
Copyright (C) OpenMps Project https://openmps.bitbucket.io/ (aokomoriuta, LWisteria@Fixstars)

OpenMps, which means all files under this repository, is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

OpenMps is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with OpenMps.  If not, see <https://www.gnu.org/licenses/>.
```

Note that OpenMps depends on [Boost C++ Libraries](https://www.boost.org/) and [ViennaCL](http://viennacl.sourceforge.net/), which is not distributed by OpenMps itself.
Therefore, the built binaries must be legal under [Boost Software License](https://www.boost.org/users/license.html), [ViennaCL's MIT License](https://github.com/viennacl/viennacl-dev/blob/master/LICENSE), and (if they have) the libraries' licenses that they depend on, not only OpenMps's GPL.
OpenMps source codes could also be same as built binaries when you distribute it with those libraries' sources.

## Usage

### Build

0. Get [Boost C++ Libraries](https://www.boost.org/) and [ViennaCL](http://viennacl.sourceforge.net/) by yourself.
	* Boost should be installed in your system and includible by OpenMps.
	* ViennaCL should be placed on `src/viennacl`. You can use `git submodule`.
0. Build codes. It is recommended to use Visual Studio.

### Execution
0. Run the program (Developers expect execution on Windows). Results will be output as CSV in "result" folder.
0. Visualize results. If you use ParaView 5.0 or later,
	1. Open "particle_****.csv"s
	1. Add "TablesToPoints" filter
		* X Column = x
		* Y Column = z
		* Z Column = x
		* Representation = Point Gaussian
		* Gaussian Radius = l_0/2
		* Shader Preset = Sphere
	1. Create new Layout (3D View)
	1. Make "TablesToPoints" visible
	1. You can see the motion of particles columns are:
		* x : horizontal value of position vector
		* z : vertical value of position vector
		* u : horizontal value of velocity vector
		* w : vertical value of velocity vector
		* p : value of pressure
		* n : value of particle number density
	1. or you can load ParaView State File "resultViewer.pvsm" for instance.

## Desclaimer
Anyone is not responsible for any damages or corrupts by this project. Download and use this project at your own risk.
Any questions or pull-resquests or other contribution are welcome. Although keeping trying to respond as soon as possible, but no one is responsible to respond it. Any responses are also not responsible by anyone.
