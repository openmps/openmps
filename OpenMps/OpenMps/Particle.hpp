#ifndef PARTICLE_INCLUDED
#define PARTICLE_INCLUDED
#include "defines.hpp"

#include "Vector.hpp"
#include "Grid.hpp"

namespace OpenMps
{
	// 粒子
	class Particle
	{
	public:
		// 粒子リスト
		typedef std::vector<Particle> List;

	protected:
		// 粒子タイプ
		typedef enum
		{
			// 非圧縮性ニュートン流体（水など）
			ParticleTypeIncompressibleNewton,

			// 壁面（位置と速度が変化しない）
			ParticleTypeWall,

			// ダミー（粒子数密度の計算にのみ対象となる）
			ParticleTypeDummy,
		} ParticleType;
		
	private:
		// 粒子タイプの数
		static const int ParticleTypeMaxCount = 3;

#ifndef PRESSURE_EXPLICIT
		// 自分を対象とした圧力方程式の係数を計算する関数ポインタの型
		typedef double(Particle::*GetPpeMatrixTargetFunc)(const Particle& source, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const;

		// 各粒子タイプで自分を対象とした圧力方程式の係数を計算する関数
		static const GetPpeMatrixTargetFunc GetPpeMatrixTargetFuncTable[ParticleTypeMaxCount];

		// 通常粒子を対象とした圧力方程式の係数を計算する
		double GetPpeMatrixTargetNormal(const Particle& source, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const
		{
			// 標準MPS法：-2D/ρλ w/n0
			double w = this->Weight(source, r_e);
			return -2*DIM/(rho*lambda) * w/n0;
		}
		
		// 自分に対する圧力方程式の係数が0である粒子を対象とした、圧力方程式の係数を計算する
		double GetPpeMatrixTargetZero(const Particle& source, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const
		{
			return 0;
		}
#endif


		// 対象の粒子へ与える粘性項を計算する関数ポインタの型
		typedef Vector(Particle::*ViscosityToFunc)(const Particle& particle_i, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const;

		// 各粒子タイプで自分を対象とした圧力方程式の係数を計算する関数
		static const ViscosityToFunc ViscosityToFuncTable[ParticleTypeMaxCount];

		// 通常粒子へ与える粘性項を計算する
		Vector ViscosityToNormal(const Particle& particle_i, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const
		{
			// 標準MPS法：ν*2D/λn0 (u_j - u_i) w（ただし自分自身からは影響を受けない）
			Vector result = (nu * 2*DIM/lambda/n_0 * particle_i.Weight(*this, r_e))*(this->u - particle_i.u);
			return result;
		}
		
		// 対象の粒子へ粘性効果を与えない粒子の与える粘性項を計算する
		Vector ViscosityToZero(const Particle& particle_i, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const
		{
			return VectorZero;
		}
		
		// 対象の粒子へ与える圧力勾配を計算する関数ポインタの型
		typedef Vector(Particle::*PressureGradientToFunc)(
			const Particle& particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP,
#endif
			const double r_e, const double dt, const double rho, const double n0) const;

		// 各粒子タイプで自分を対象とした圧力方程式の係数を計算する関数
		static const PressureGradientToFunc PressureGradientToFuncTable[ParticleTypeMaxCount];

		// 通常粒子へ与える圧力勾配を計算する
		Vector PressureGradientToNormal(
			const Particle& particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP,
#endif
			const double r_e, const double dt, const double rho, const double n0) const
		{
			namespace ublas = boost::numeric::ublas;

			auto dx = this->x - particle_i.x;
			auto r2 = ublas::inner_prod(dx, dx);

			// 標準MPS法：-Δt/ρ D/n_0 (p_j + p_i)/r^2 w * dx（ただし自分自身からは影響を受けない）
#ifdef PRESSURE_GRADIENT_MIDPOINT
			auto result = -(dt/rho * DIM/n0 * (this->p + particle_i.p)/r2 * particle_i.Weight(*this, r_e));
#else
			auto result = -(dt/rho * DIM/n0 * (this->p - minP        )/r2 * particle_i.Weight(*this, r_e));
#endif
			return (r2 == 0 ? 0 : result) * dx;
		}
		
		// 対象の粒子へ圧力勾配を与えない粒子の圧力勾配を計算する
		Vector PressureGradientToZero(
			const Particle& particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP,
#endif
			const double r_e, const double dt, const double rho, const double n0) const
		{
			return VectorZero;
		}


		// 粒子を加速（速度を変更）する関数ポインタの型
		typedef void(Particle::*AccelerateFunc)(const Vector& du);

		// 各粒子タイプで粒子を加速（速度を変更）する関数
		static const AccelerateFunc AccelerateFuncTable[ParticleTypeMaxCount];

		// 通常粒子を加速（速度を変更）する
		void AccelerateNormal(const Vector& du)
		{
			u += du;
		}
		
		// 移動しない粒子を加速（速度を変更）する
		void AccelerateZero(const Vector&)
		{
			// 動かさない
		}


		// 粒子を移動（位置を変更）する関数ポインタの型
		typedef void(Particle::*MoveFunc)(const Vector& dx);

		// 各粒子タイプで粒子を移動（位置を変更）する関数
		static const MoveFunc MoveFuncTable[ParticleTypeMaxCount];

		// 通常粒子を移動（位置を変更）する
		void MoveNormal(const Vector& dx)
		{
			x += dx;
		}
		
		// 移動しない粒子を移動（位置を変更）する
		void MoveZero(const Vector&)
		{
			// 動かさない
		}


		// 粘性項を計算する関数ポインタの型
		typedef Vector(Particle::*GetViscosityFunc)(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const;

		// 各粒子タイプで粘性項を計算する関数
		static const GetViscosityFunc GetViscosityFuncTable[ParticleTypeMaxCount];

		// 通常粒子の粘性項を計算するする
		Vector GetViscosityNormal(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const;
		
		// 移動しない粒子の粘性項を計算する
		Vector GetViscosityZero(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const
		{
			return VectorZero;
		}


		// 重み関数を計算する関数ポインタの型
		typedef double(Particle::*WeightFunc)(const Particle& target, const double r_e) const;

		// 各粒子タイプで重み関数を計算する関数
		static const WeightFunc WeightFuncTable[ParticleTypeMaxCount];

		// 通常粒子の重み関数を計算する
		double WeightNormal(const Particle& target, const double r_e) const
		{
			return target.WeightTarget(*this, r_e);
		}
		
		// 重み関数を計算しない粒子の重み関数を計算する
		double WeightZero(const Particle& target, const double r_e) const
		{
			return 0;
		}

		
		// 粘性項を計算する関数ポインタの型
		typedef void(Particle::*UpdateNeighborDensityFunc)(const Particle::List& particles, const Grid& grid, const double r_e);

		// 各粒子タイプで粘性項を計算する関数
		static const UpdateNeighborDensityFunc UpdateNeighborDensityFuncTable[ParticleTypeMaxCount];

		// 通常粒子の粘性項を計算する
		void UpdateNeighborDensityNormal(const Particle::List& particles, const Grid& grid, const double r_e);
		
		// 移動しない粒子の粘性項を計算する
		void UpdateNeighborDensityZero(const Particle::List& particles, const Grid& grid, const double r_e)
		{
			// 計算しない
			n = 0;
		}

		
#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する関数ポインタの型
		typedef Vector(Particle::*GetCorrectionByTooNearFunc)(const Particle::List& particles, const Grid& grid, const double r_e, const double rho, const double tooNearRatio, const double tooNearCoefficient) const;

		// 各粒子タイプで過剰接近粒子からの速度補正量を計算する関数
		static const GetCorrectionByTooNearFunc GetCorrectionByTooNearFuncTable[ParticleTypeMaxCount];

		// 通常粒子の過剰接近粒子からの速度補正量を計算する
		Vector GetCorrectionByTooNearNormal(const Particle::List& particles, const Grid& grid, const double r_e, const double rho, const double tooNearRatio, const double tooNearCoefficient) const;
		
		// 移動しない粒子の過剰接近粒子からの速度補正量を計算する
		Vector GetCorrectionByTooNearZero(const Particle::List& particles, const Grid& grid, const double r_e, const double rho, const double tooNearRatio, const double tooNearCoefficient) const
		{
			return VectorZero;
		}
#endif

		
#ifdef PRESSURE_EXPLICIT
		// 圧力を計算する関数ポインタの型
		typedef void(Particle::*UpdatePressureFunc)(const double c, const double rho0, const double n0);

		// 各粒子タイプで圧力を計算する関数
		static const UpdatePressureFunc UpdatePressureFuncTable[ParticleTypeMaxCount];

		// 通常粒子の圧力を計算する
		void UpdatePressureNormal(const double c, const double rho0, const double n0)
		{
			// 仮想的な密度：ρ0/n0 * n
			auto rho = rho0/n0 * n;

			// 圧力の計算：c^2 (ρ-ρ0)（基準密度以下なら圧力は発生しない）
			p = (rho <= rho0) ? 0 : c*c*(rho - rho0);
		}
		
		// 圧力を持たない粒子の圧力を計算する
		void UpdatePressureZero(const double c, const double rho0, const double n0)
		{
			// 計算しない
			p = 0;
		}
#else


		// 圧力方程式の生成項を計算する関数ポインタの型
		typedef double(Particle::*GetPpeSourceFunc)(const double n0, const double dt, const double surfaceRatio) const;

		// 各粒子タイプで圧力方程式の生成項を計算する関数
		static const GetPpeSourceFunc GetPpeSourceFuncTable[ParticleTypeMaxCount];

		// 通常粒子の圧力方程式の生成項を計算する
		double GetPpeSourceNormal(const double n0, const double dt, const double surfaceRatio) const
		{
			// 自由表面の場合は0
			return IsSurface(n0, surfaceRatio) ? 0
				// 標準MPS法：b_i = 1/dt^2 * (n_i - n0)/n0
				: (n - n0)/n0 /(dt*dt);
		}
		
		// 圧力を持たない粒子の圧力方程式の生成項を計算する
		double GetPpeSourceZero(const double c, const double rho0, const double n0) const
		{
			// 計算しない
			return 0;
		}


		// 圧力方程式の係数を計算する関数ポインタの型
		typedef double(Particle::*GetPpeMatrixFunc)(const Particle& target, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const;

		// 各粒子タイプで圧力方程式の係数を計算する関数
		static const GetPpeMatrixFunc GetPpeMatrixFuncTable[ParticleTypeMaxCount];

		// 通常粒子の圧力方程式の係数を計算する
		double GetPpeMatrixNormal(const Particle& target, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const
		{
			// 自由表面の場合は0
			return IsSurface(n0, surfaceRatio) ? 0
				: target.GetPpeMatrixTarget(*this, n0, r_e, lambda, rho, surfaceRatio);
		}
		
		// 圧力を持たない粒子の圧力方程式の係数を計算する
		double GetPpeMatrixZero(const Particle& target, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const
		{
			// 計算しない
			return 0;
		}
#endif

		// 圧力勾配を計算する関数ポインタの型
		typedef Vector(Particle::*GetPressureGradientFunc)(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0) const;

		// 各粒子タイプで圧力勾配を計算する関数
		static const GetPressureGradientFunc GetPressureGradientFuncTable[ParticleTypeMaxCount];

		// 通常粒子の圧力勾配を計算する
		Vector GetPressureGradientNormal(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0) const;
		
		// 移動しない粒子の圧力勾配を計算する
		Vector GetPressureGradientZero(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0) const
		{
			// 計算しない
			return VectorZero;
		}

	protected:
		// 位置ベクトル
		Vector x;

		// 速度ベクトル
		Vector u;

		// 圧力
		double p;

		// 粒子数密度
		double n;

		// 粒子タイプ
		const ParticleType type;

		// 自分を対象とした重み関数を計算する
		// @param source 基準とする粒子
		// @param r_e 影響半径
		double WeightTarget(const Particle& source, const double r_e) const
		{
			namespace ublas = boost::numeric::ublas;

			// 2粒子間の距離から重み関数の値を返す
			auto r = ublas::norm_2(source.x - this->x);
			return Particle::Weight(r, r_e);
		}

#ifndef PRESSURE_EXPLICIT
		// 自分を対象とした圧力方程式の係数を計算する
		// @param source 基準とする粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param rho 密度
		// @param surfaceRatio 自由表面の判定係数（基準粒子数密度からのずれがこの割合以下なら自由表面と判定される）
		double GetPpeMatrixTarget(const Particle& source, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const
		{
			return (this->*(Particle::GetPpeMatrixTargetFuncTable[type]))(source, n0, r_e, lambda, rho, surfaceRatio);
		}
#endif

		// 対象の粒子へ与える粘性項を計算する
		// @param particle_i 対象の粒子粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		Vector ViscosityTo(const Particle& particle_i, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const
		{
			return (this->*(Particle::ViscosityToFuncTable[type]))(particle_i, n_0, r_e, lambda, nu, dt);
		}

		// 対象の粒子へ与える圧力勾配を計算する
		// @param particle_i 対象の粒子
#ifndef PRESSURE_GRADIENT_MIDPOINT
		// @param minP 計算で使用する自分の圧力（周囲の最小圧力）
#endif
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		Vector PressureGradientTo(
			const Particle& particle_i, 
#ifndef PRESSURE_GRADIENT_MIDPOINT
			const double minP, 
#endif
			const double r_e, const double dt, const double rho, const double n0) const
		{
			return (this->*(Particle::PressureGradientToFuncTable[type]))(particle_i,
#ifndef PRESSURE_GRADIENT_MIDPOINT
				minP,
#endif
				r_e, dt, rho, n0);
		}

		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param u 速度ベクトルの水平方向成分
		// @param w 速度ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		Particle(const ParticleType type, const double x, const double z, const double u, const double w, const double p, const double n);

	public:

		// 距離から重み関数を計算する
		// @param r 距離
		// @param r_e 影響半径
		static double Weight(const double r, const double r_e)
		{
			// 影響半径内ならr_e/r-1を返す（ただし距離0の場合は0）
			return ((0 < r) && (r < r_e)) ? (r_e/r - 1) : 0;
		}

		// 粒子を加速（速度を変更）する
		// @param du 速度の変化量
		void Accelerate(const Vector& du)
		{
			(this->*(Particle::AccelerateFuncTable[type]))(du);
		}

		// 粒子を移動（位置を変更）する
		// @param dx 位置の変化量
		void Move(const Vector& dx)
		{
			(this->*(Particle::MoveFuncTable[type]))(dx);
		}

		// 粘性項を計算する
		// @param particles 粒子リスト
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		Vector GetViscosity(const Particle::List& particles, const Grid& grid, const double n_0, const double r_e, const double lambda, const double nu, const double dt) const
		{
			return (this->*(Particle::GetViscosityFuncTable[type]))(particles, grid, n_0, r_e, lambda, nu, dt);
		}

		// 重み関数を計算する
		// @param target 計算相手の粒子
		// @param r_e 影響半径
		double Weight(const Particle& target, const double r_e) const
		{
			return (this->*(Particle::WeightFuncTable[type]))(target, r_e);
		}

		// 粒子数密度を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		void UpdateNeighborDensity(const Particle::List& particles, const Grid& grid, const double r_e)
		{
			return (this->*(Particle::UpdateNeighborDensityFuncTable[type]))(particles, grid, r_e);
		}

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		// @param rho 密度
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		Vector GetCorrectionByTooNear(const Particle::List& particles,const Grid& grid, const double r_e, const double rho, const double tooNearRatio, const double tooNearCoefficient) const
		{
			return (this->*(Particle::GetCorrectionByTooNearFuncTable[type]))(particles, grid, r_e, rho, tooNearRatio, tooNearCoefficient);
		}
#endif

#ifdef PRESSURE_EXPLICIT
		// 圧力を計算する
		// @param c 音速
		// @param rho0 （基準）密度
		// @param n0 基準粒子数密度
		void UpdatePressure(const double c, const double rho0, const double n0)
		{
			return (this->*(Particle::UpdatePressureFuncTable[type]))(c, rho0, n0);
		}
#else
		// 圧力方程式の生成項を計算する
		// @param n0 基準粒子数密度
		// @param dt 時間刻み
		// @param surfaceRatio 自由表面の判定係数（基準粒子数密度からのずれがこの割合以下なら自由表面と判定される）
		double GetPpeSource(const double n0, const double dt, const double surfaceRatio) const
		{
			return (this->*(Particle::GetPpeSourceFuncTable[type]))(n0, dt, surfaceRatio);
		}

		// 圧力方程式の係数を計算する
		// @param particle 対象粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param rho 密度
		double GetPpeMatrix(const Particle& target, const double n0, const double r_e, const double lambda, const double rho, const double surfaceRatio) const
		{
			return (this->*(Particle::GetPpeMatrixFuncTable[type]))(target, n0, r_e, lambda, rho, surfaceRatio);
		} 
#endif

		// 圧力勾配を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 基準粒子数密度
		Vector GetPressureGradient(const Particle::List& particles, const Grid& grid, const double r_e, const double dt, const double rho, const double n0) const
		{	
			return (this->*(Particle::GetPressureGradientFuncTable[type]))(particles, grid, r_e, dt, rho, n0);
		}

		////////////////
		// プロパティ //
		////////////////

		// 位置ベクトルの水平方向成分を取得する
		inline double X() const
		{
			return x[0];
		}

		// 位置ベクトルの鉛直成分を取得する
		inline double Z() const
		{
			return x[1];
		}

		// 速度ベクトルの水平方向成分を取得する
		inline double U() const
		{
			return u[0];
		}

		// 速度ベクトルの鉛直方向成分を取得する
		inline double W() const
		{
			return u[1];
		}

		// 圧力を取得する
		inline double P() const
		{
			return p;
		}

		// 圧力を設定する
		// @param value 設定値
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		inline void P(const double value, const double n0, const double surfaceRatio)
		{
			// 負圧であったり自由表面の場合は圧力0
			p = ((value < 0) || IsSurface(n0, surfaceRatio)) ? 0 : value;
		}

		// 粒子数密度を取得する
		inline double N() const
		{
			return n;
		}

		// 圧力を取得する
		inline ParticleType Type() const
		{
			return type;
		}

		// 位置ベクトルを取得する
		inline Vector VectorX() const
		{
			return x;
		}

		// 速度ベクトルを取得する
		inline Vector VectorU() const
		{
			return u;
		}

		// 自由表面かどうかの判定
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		inline bool IsSurface(const double n0, const double surfaceRatio) const
		{
			return n/n0 < surfaceRatio;
		}
	};
	

	// 非圧縮性ニュートン流体（水など）
	class ParticleIncompressibleNewton : public Particle
	{
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param u 速度ベクトルの水平方向成分
		// @param w 速度ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		ParticleIncompressibleNewton(const double x, const double z, const double u, const double w, const double p, const double n)
			: Particle(ParticleTypeIncompressibleNewton, x, z, u, w, p, n)
		{
		}
	};
	
	// 壁粒子（位置と速度が変化しない）
	class ParticleWall : public Particle
	{
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		ParticleWall(const double x, const double z, const double p, const double n)
			: Particle(ParticleTypeWall, x, z, 0, 0, p, n)
		{
		}
	};

	// ダミー粒子（粒子数密度の計算にのみ対象となる）
	class ParticleDummy : public Particle
	{
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		ParticleDummy(const double x, const double z)
			: Particle(ParticleTypeDummy, x, z, 0, 0, 0, 0)
		{
		}
	};
}
#endif