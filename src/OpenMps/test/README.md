## Usage

There are test programs for OpenMps in this directory.
It is recommended to use Visual Studio.

### Build

0. You should build a test library.
	1. Get [Googletest](https://github.com/google/googletest) by yourself.
		* Googletest should be placed on `src/googletest`. You can use `git submodule`.
	1. Generate scripts to build the library with `cmake`.
		* For Visual Studio, You can generate project files(`ALL_BUILD.vcxproj, ...`).
	1. Build the library. 

0. Build codes in `src/OpenMps/test`.

### Execution

0. Run the program (Developers expect execution on Windows). Results will be output in the console window.
