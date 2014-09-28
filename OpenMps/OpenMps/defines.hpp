#ifndef DEFINE_INCLUDED
#define DEFINE_INCLUDED

// 次元数
static const int DIM = 2;

/////////////////////////////////////////////
// 以下は計算手法についての選択肢          //
// 何も指定しなければ標準MPS法が採用される //
/////////////////////////////////////////////

// 過剰接近粒子の補正を導入する
// 　長所：計算が安定する（発散しにくくなる）
// 　短所：運動量が不正確になる
#define MODIFY_TOO_NEAR

// 圧力勾配項の計算で、圧力を2粒子間の中点で評価する
// 　長所：作用・反作用を満たさない非対称な力がなくなる
// 　短所：なし（不安定化？）
#define PRESSURE_GRADIENT_MIDPOINT

// 圧力の計算に陽解法を用いる場合に指定する
// 　長所：1ステップの計算が陰解法に比べて非常に高速
// 　短所：時間刻みが小さくなる・弱圧縮を許容する
// #define PRESSURE_EXPLICIT

/**************************************************************/

//////////////////////////////////////////////////////////////
// 計算の高速化についての選択肢                             //
// 何も指定しなければ並列化されず、boostのuBLASが採用される //
//////////////////////////////////////////////////////////////

// 行列計算にViennaCLを使用する。
#define USE_VIENNACL

// OpenCLを使用する。
//#define USE_OPENCL

// CUDAを使用する。
//#define USE_CUDA

/**************************************************************/
// 以下、自動設定（手動で変更しないこと）
#ifdef USE_VIENNACL
	#ifdef USE_OPENCL
		// ViennCLでOpenCLを使用する
		#define VIENNACL_WITH_OPENCL
	#else
		#ifdef USE_CUDA
			// ViennCLでCUDAを使用する（※まだ動かない）
			#define VIENNACL_WITH_CUDA
			#error CUDAでのViennaCLは未実装です
		#else
			#ifdef _OPENMP
				// ViennaCLでOpenMPを使用する
				#define VIENNACL_WITH_OPENMP
			#endif
		#endif
	#endif
#else
	#ifdef USE_OPENCL
		// APPMLを使用する（※まだ動かない）
		#define USE_APPML
		#error APPMLは未実装です
	#else
		#ifdef USE_CUDA
			// CUBLAS&CUSPARSEを使用する（※まだ動かない）
			#define USE_CUBLAS
			#error CUBLAS&CUSPARSEは未実装です
		#endif
	#endif
#endif
#endif