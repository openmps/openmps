# OpenMps

author (attribution)
: 青子守歌 (aokomoriuta)

URL
: https://github.com/aokomoriuta

This project contains an implemention of Moving Particle Semi-implicit (MPS) method.
MPS method is one of the most popular Particle Method to solve Continuum Dynamics (e.g. Fluid Dynamics). See [Wikipedia](https://en.wikipedia.org/wiki/Moving_Particle_Semi-implicit_Method) for more information.

This project is multi-licensed.
You can select the license of your choice from as following:

* [Creative Commons Attribution-ShareAlike 3.0 Unported](http://creativecommons.org/licenses/by-sa/3.0/)
* [Creative Commons Attribution-Noncommercial 3.0 Unported](http://creativecommons.org/licenses/by-nc/3.0/)
* [GNU General Public License v3.0 or later](http://www.gnu.org/licenses/gpl.html)

## Usage
### Initialization
When you check out this project first, you need to

0. get [boost](http://www.boost.org) and uncompress it into "boost" folder.
0. build boost by execution "bootstrap" and "b2"

### Build

0. Build codes. It is recommended to use Visual Studio 2015.

### Execution
0. Run the program (Developers expect execution on Windows). Results will be output as CSV in "result" folder.
0. Visualize results. If you use ParaView,
    1. Enable PointSprite plugin on [Tools]-[Plugin Manager]-[Local Plugins]-[PointSprite_Plugin]-[Load Selected]
	1. Open "particle_****.csv"s
	1. Add "TablesToPoints" filter
		* X Column = x
		* Y Column = z
		* Z Column = x
		* 2D Points = ON
	1. Create new Layout (3D View)
	1. Make "TablesToPoints" visible
	1. You can see the motion of particles! csv columns are:
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

## License

### FDPS
Copyright (c) Iwasawa et al. (2015 in prep) Tanikawa et al. (2016 in prep)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
