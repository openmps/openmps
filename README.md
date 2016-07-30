# OpenMps

This project contains an implemention of Moving Particle Semi-implicit (MPS) method.
MPS method is one of the most popular Particle Method to solve Continuum Dynamics (e.g. Fluid Dynamics). See [Wikipedia](https://en.wikipedia.org/wiki/Moving_Particle_Semi-implicit_Method) for more information.

![screenshot](https://bytebucket.org/OpenMps/openmps.bitbucket.org/raw/master/img/icon.PNG)

## License

This project is multi-licensed.
You can select the license of your choice from as following:

* [Creative Commons Attribution-ShareAlike 3.0 Unported](http://creativecommons.org/licenses/by-sa/3.0/)
* [Creative Commons Attribution-Noncommercial 3.0 Unported](http://creativecommons.org/licenses/by-nc/3.0/)
* [GNU General Public License v3.0 or later](http://www.gnu.org/licenses/gpl.html)

author (attribution)
: 青子守歌 (aokomoriuta)

URL
: https://github.com/aokomoriuta

## Usage

### Build

0. Build codes. It is recommended to use Visual Studio 2015.

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
Anyone including me (author:aokomoriuta) is not responsible for any damages or corrupts by this project. Download and use this project at your own risk.
Any questions or pull-resquests or other contribution are welcome. Although keeping trying to respond as soon as possible, but no one is responsible to respond it. Any responses are also not responsible by anyone including me.
