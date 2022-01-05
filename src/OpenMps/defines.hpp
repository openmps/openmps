﻿#ifndef DEFINE_INCLUDED
#define DEFINE_INCLUDED

/////////////////////////////////////////////
// 以下は計算手法についての選択肢          //
// 何も指定しなければ標準MPS法が採用される //
/////////////////////////////////////////////

// 3次元にする
// ※用意はしているが現時点では非推奨
// #define DIM3

// 剛体球の衝突を模した過剰接近粒子の補正を導入する
// 　長所：計算が安定する（発散しにくくなる）
// 　短所：運動量が不正確になる
// #define ARTIFICIAL_COLLISION_FORCE

// 圧力勾配項の計算で、圧力を2粒子間の中点で評価する
// 　長所：作用・反作用を満たさない非対称な力がなくなる
// 　短所：なし（不安定化？）
#define PRESSURE_GRADIENT_MIDPOINT

// 圧力の計算に陽解法を用いる場合に指定する
// 　長所：1ステップの計算が陰解法に比べて非常に高速
// 　短所：時間刻みが小さくなる・弱圧縮を許容する
// #define PRESSURE_EXPLICIT

// HS法（高精度生成項）, Khayyer and Gotoh (2009)
//   長所：高精度化
//   短所：計算負荷の微増
#define MPS_HS

// HL法（高精度ラプラシアン）, Khayyer and Gotoh (2010)
//   長所：高精度化
//   短所：なし
//#define MPS_HL

// 1次精度ラプラシアン（λへの置き換えを行わないモデル）, 後藤(2018)『粒子法』(3.151)
#define MPS_1L

// ECS法（誤差修正項）, Khayyer and Gotoh (2011)
//   長所：高精度化・安定化
//   短所：計算負荷・メモリ使用量の微増
#define MPS_ECS

// GC法（勾配修正行列）, Khayyer and Gotoh (2011)
//   長所：安定化
//   短所：計算負荷の微増、作用・反作用を満たさない
//#define MPS_GC

// DS法（動的人工斥力）, Tsuruta et al. (2013)
//   長所：安定化
//   短所：運動量がやや不正確になる（剛体球衝突のモデルよりは低減されている）
#define MPS_DS

// SPP法（自由表面仮想粒子）, Tsuruta et al. (2015)
//   長所：非物理的な空隙の除去、自由表面の追跡性高精度化
//   短所：なし
#define MPS_SPP

/**************************************************************/

//////////////////////////////////////////////////////////////
// 計算の高速化についての選択肢                             //
// 何も指定しなければ並列化されず、boostのuBLASが採用される //
//////////////////////////////////////////////////////////////

// 行列計算にViennaCLを使用する。
#define USE_VIENNACL

/**************************************************************/
// 以下、自動設定（手動で変更しないこと）
#ifdef USE_VIENNACL
	#ifdef _OPENMP
		// ViennaCLでOpenMPを使用する
		#define VIENNACL_WITH_OPENMP
	#endif
#endif

#include <cstddef>

namespace { namespace OpenMps
{
#ifdef DIM3
	static constexpr std::size_t DIM = 3;
	static constexpr std::size_t AXIS_X = 0;
	static constexpr std::size_t AXIS_Y = 1;
	static constexpr std::size_t AXIS_Z = 2;
#else
	static constexpr std::size_t DIM = 2;
	static constexpr std::size_t AXIS_X = 0;
	static constexpr std::size_t AXIS_Z = 1;
#endif
}}

#endif
