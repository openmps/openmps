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

	protected:
		// 位置ベクトル
		Vector x;

		// 速度ベクトル
		Vector u;

		// 圧力
		double p;

		// 粒子数密度
		double n;

		// 自分を対象とした重み関数を計算する
		// @param source 基準とする粒子
		// @param r_e 影響半径
		inline virtual double WeightTarget(const Particle& source, const double& r_e) const
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
		inline virtual double MatrixTarget(const Particle& source, const double& n0, const double& r_e, const double& lambda, const double& rho, const double& surfaceRatio) const
		{
			// 標準MPS法：-2D/ρλ w/n0
			double w = this->Weight(source, r_e);
			return -2*DIM/(rho*lambda) * w/n0;
		}
#endif

		// 対象の粒子へ与える粘性項を計算する
		// @param particle_i 対象の粒子粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		inline virtual Vector ViscosityTo(const Particle& particle_i, const double& n_0, const double& r_e, const double& lambda, const double& nu, const double& dt) const
		{			
			// 標準MPS法：ν*2D/λn0 (u_j - u_i) w（ただし自分自身からは影響を受けない）
			Vector result = (nu * 2*DIM/lambda/n_0 * particle_i.Weight(*this, r_e))*(this->u - particle_i.u);
			return result;
		}

		
#ifdef PRESSURE_GRADIENT_MIDPOINT
		// 対象の粒子へ与える圧力勾配を計算する
		// @param particle_i 対象の粒子
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		inline virtual Vector PressureGradientTo(const Particle& particle_i, const double& r_e, const double& dt, const double& rho, const double& n0)
		{
			namespace ublas = boost::numeric::ublas;

			// 標準MPS法：-Δt/ρ D/n_0 (p_j + p_i)/r^2 w * dx（ただし自分自身からは影響を受けない）
			auto dx = this->x - particle_i.x;
			auto r2 = ublas::inner_prod(dx, dx);
			auto result = -(dt/rho * DIM/n0 * (this->p + particle_i.p)/r2 * particle_i.Weight(*this, r_e));
			return (r2 == 0 ? 0 : result) * dx;
		}
#else
		// 対象の粒子へ与える圧力勾配を計算する
		// @param particle_i 対象の粒子
		// @param minP 計算で使用する自分の圧力（周囲の最小圧力）
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		inline virtual Vector PressureGradientTo(const Particle& particle_i, const double& minP, const double& r_e, const double& dt, const double& rho, const double& n0)
		{
			namespace ublas = boost::numeric::ublas;

			// 標準MPS法：-Δt/ρ D/n_0 (p_j - p_i)/r^2 w * dx（ただし自分自身からは影響を受けない）
			auto dx = this->x - particle_i.x;
			auto r2 = ublas::inner_prod(dx, dx);
			auto result = -(dt/rho * DIM/n0 * (this->p - minP)/r2 * particle_i.Weight(*this, r_e));
			return (r2 == 0 ? 0 : result) * dx;
		}
#endif

		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param u 速度ベクトルの水平方向成分
		// @param w 速度ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		Particle(const double& x, const double& z, const double& u, const double& w, const double& p, const double& n);

	public:
		// 粒子へのポインタ
		typedef std::shared_ptr<Particle> Ptr;
		
		// 粒子リスト
		typedef std::vector<Particle::Ptr> List;

		// 距離から重み関数を計算する
		// @param r 距離
		// @param r_e 影響半径
		inline static double Weight(const double& r, const double& r_e)
		{
			// 影響半径内ならr_e/r-1を返す（ただし距離0の場合は0）
			return ((0 < r) && (r < r_e)) ? (r_e/r - 1) : 0;
		}

		// 粒子を加速（速度を変更）する
		// @param du 速度の変化量
		inline virtual void Accelerate(const Vector& du)
		{
			u += du;
		}

		// 粒子の移動（位置を変更）する
		// @param dx 位置の変化量
		inline virtual void Move(const Vector& dx)
		{
			x += dx;
		}

		// 粘性項を計算する
		// @param particles 粒子リスト
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		virtual Vector GetViscosity(const Particle::List& particles, const Grid& grid, const double& n_0, const double& r_e, const double& lambda, const double& nu, const double& dt) const;


		// 重み関数を計算する
		// @param target 計算相手の粒子
		// @param r_e 影響半径
		inline virtual double Weight(const Particle& target, const double& r_e) const
		{
			return target.WeightTarget(*this, r_e);
		}

		// 粒子数密度を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		virtual void UpdateNeighborDensity(const Particle::List& particles, const Grid& grid, const double& r_e);

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		// @param rho 密度
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		virtual Vector GetCorrectionByTooNear(const Particle::List& particles,const Grid& grid, const double& r_e, const double& rho, const double& tooNearRatio, const double& tooNearCoefficient) const;
#endif

#ifdef PRESSURE_EXPLICIT
		// 圧力を計算する
		// @param c 音速
		// @param rho0 （基準）密度
		// @param n0 基準粒子数密度
		inline virtual void UpdatePressure(const double& c, const double& rho0, const double& n0)
		{
			// 仮想的な密度：ρ0/n0 * n
			auto rho = rho0/n0 * n;

			// 圧力の計算：c^2 (ρ-ρ0)（基準密度以下なら圧力は発生しない）
			p = (rho <= rho0) ? 0 : c*c*(rho - rho0);
		}
#else
		// 圧力方程式の生成項を計算する
		// @param n0 基準粒子数密度
		// @param dt 時間刻み
		// @param surfaceRatio 自由表面の判定係数（基準粒子数密度からのずれがこの割合以下なら自由表面と判定される）
		inline virtual double Source(const double& n0, const double& dt, const double& surfaceRatio) const
		{
			// 自由表面の場合は0
			return IsSurface(n0, surfaceRatio) ? 0
				// 標準MPS法：b_i = 1/dt^2 * (n_i - n0)/n0
				: (n - n0)/n0 /(dt*dt);
		}

		// 圧力方程式の係数を計算する
		// @param particle 対象粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param rho 密度
		inline virtual double Matrix(const Particle& target, const double& n0, const double& r_e, const double& lambda, const double& rho, const double& surfaceRatio) const
		{
			// 自由表面の場合は0
			return IsSurface(n0, surfaceRatio) ? 0
				: target.MatrixTarget(*this, n0, r_e, lambda, rho, surfaceRatio);
		} 
#endif

		// 圧力勾配を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 基準粒子数密度
		virtual Vector GetPressureGradient(const Particle::List& particles, const Grid& grid, const double& r_e, const double& dt, const double& rho, const double& n0) const;

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
		inline void P(const double value, const double& n0, const double& surfaceRatio)
		{
			// 負圧であったり自由表面の場合は圧力0
			p = ((value < 0) || IsSurface(n0, surfaceRatio)) ? 0 : value;
		}

		// 粒子数密度を取得する
		inline double N() const
		{
			return n;
		}

		// 位置ベクトルを取得する
		inline const Vector& VectorX()
		{
			return x;
		}

		// 速度ベクトルを取得する
		inline const Vector& VectorU()
		{
			return u;
		}

		// 自由表面かどうかの判定
		// @param n0 基準粒子数密度
		// @param surfaceRatio 自由表面判定係数
		inline bool IsSurface(const double& n0, const double& surfaceRatio) const
		{
			return n/n0 < surfaceRatio;
		}
	};
	

	// 非圧縮性ニュートン流体（水など）
	class ParticleIncompressibleNewton : public Particle
	{
	protected:
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param u 速度ベクトルの水平方向成分
		// @param w 速度ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		ParticleIncompressibleNewton(const double& x, const double& z, const double& u, const double& w, const double& p, const double& n)
			: Particle(x, z, u, w, p, n)
		{
		}
	};
	
	// 壁粒子（位置と速度が変化しない）
	class ParticleWall : public Particle
	{
	protected:
	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		// @param p 圧力
		// @param n 粒子数密度
		ParticleWall(const double& x, const double& z, const double& p, const double& n)
			: Particle(x, z, 0, 0, p, n)
		{
		}

		// 粒子を加速（速度を変更）する
		// @param du 速度の変化量
		inline virtual void Accelerate(const Vector& du)
		{
			// 動かさない
		}

		// 粒子の移動（位置を変更）する
		// @param dx 位置の変化量
		inline virtual void Move(const Vector& dx)
		{
			// 動かさない
		}

		// 粘性項を計算する
		// @param particles 粒子リスト
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		virtual Vector GetViscosity(const Particle::List& particles, const Grid& grid, const double& n_0, const double& r_e, const double& lambda, const double& nu) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		// @param rho 密度
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		virtual Vector GetCorrectionByTooNear(const Particle::List& particles,const Grid& grid, const double& r_e, const double& rho, const double& tooNearRatio, const double& tooNearCoefficient) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}
#endif

		// 圧力勾配を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		virtual Vector GetPressureGradient(const Particle::List& particles, const Grid& grid, const double& r_e, const double& dt, const double& rho, const double& n0) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}
	};

	// ダミー粒子（粒子数密度の計算にのみ対象となる）
	class ParticleDummy : public Particle
	{
	protected:
#ifndef PRESSURE_EXPLICIT
		// 自分を対象とした圧力方程式の係数を計算する
		// @param source 基準とする粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param rho 密度
		inline virtual double MatrixTarget(const Particle& source, const double& n0, const double& r_e, const double& lambda, const double& rho, const double& surfaceRatio) const
		{
			// 相手基準の係数は常に0
			return 0;
		}
#endif


		// 対象の粒子から受ける粘性項を計算する
		// @param particle 対象の粒子粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		inline virtual Vector ViscosityTo(const Particle& particle, const double& n_0, const double& r_e, const double& lambda, const double& nu, const double& dt) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}

#ifdef MODIFY_TOO_NEAR
		// 過剰接近粒子からの速度補正量を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		// @param rho 密度
		// @param tooNearRatio 過剰接近粒子と判定される距離（初期粒子間距離との比）
		// @param tooNearCoeffcient 過剰接近粒子から受ける修正量の係数
		virtual Vector GetCorrectionByTooNear(const Particle::List& particles,const Grid& grid, const double& r_e, const double& rho, const double& tooNearRatio, const double& tooNearCoefficient) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}
#endif

#ifdef PRESSURE_GRADIENT_MIDPOINT
		// 対象の粒子へ与える圧力勾配を計算する
		// @param particle_i 対象の粒子
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		inline virtual Vector PressureGradientTo(const Particle& particle_i, const double& r_e, const double& dt, const double& rho, const double& n0)
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}
#else
		// 対象の粒子から受ける圧力勾配を計算する
		// @param particle 対象の粒子
		// @param thisP 計算で使用する自分の圧力（周囲の最小圧力）
		// @param r_e 影響半径
		// @param dt 時間刻み
		// @param rho 密度
		// @param n0 粒子数密度
		inline virtual Vector PressureGradientTo(const Particle& particle, const double& thisP, const double& r_e, const double& dt, const double& rho, const double& n0)
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}
#endif

	public:
		// @param type 粒子タイプ
		// @param x 位置ベクトルの水平方向成分
		// @param z 位置ベクトルの鉛直方向成分
		ParticleDummy(const double& x, const double& z)
			: Particle(x, z, 0, 0, 0, 0)
		{
		}

		// 粒子を加速（速度を変更）する
		// @param du 速度の変化量
		inline virtual void Accelerate(const Vector& du)
		{
			// 動かさない
		}

		// 粒子の移動（位置を変更）する
		// @param dx 位置の変化量
		inline virtual void Move(const Vector& dx)
		{
			// 動かさない
		}

		// 重み関数を計算する
		// @param target 計算相手の粒子
		// @param r_e 影響半径
		inline virtual double Weight(const Particle& target, const double& r_e) const
		{
			// 常に0
			return 0;
		}
		// 粒子数密度を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		virtual void UpdateNeighborDensity(const Particle::List& particles, const Grid& grid, const double& r_e)
		{
			// 計算しない
			n = 0;
		}

#ifdef PRESSURE_EXPLICIT
		// 圧力を計算する
		// @param c 音速
		// @param rho0 （基準）密度
		// @param n0 基準粒子数密度
		inline virtual void UpdatePressure(const double& c, const double& rho0, const double& n0)
		{
			// 常に0
			p = 0;
		}
#else
		// 圧力方程式の生成項を計算する
		// @param n0 基準粒子数密度
		// @param dt 時間刻み
		// @param surfaceRatio 自由表面の判定係数（基準粒子数密度からのずれがこの割合以下なら自由表面と判定される）
		inline virtual double Source(const double& n0, const double& dt, const double& surfaceRatio) const
		{
			// 生成項は0
			return 0;
		}
		
		// 圧力方程式の係数を計算する
		// @param particle 対象粒子
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param rho 密度
		inline virtual double Matrix(const Particle& target, const double& n0, const double& r_e, const double& lambda, const double& rho, const double& surfaceRatio) const
		{
			// 自分基準の係数は常に0
			return 0;
		}
#endif

		// 粘性項を計算する
		// @param particles 粒子リスト
		// @@aram n_0 基準粒子数密度
		// @param r_e 影響半径
		// @param lambda 拡散モデル係数λ
		// @param nu 粘性係数
		virtual Vector GetViscosity(const Particle::List& particles, const Grid& grid, const double& n_0, const double& r_e, const double& lambda, const double& nu) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}


		// 圧力勾配を計算する
		// @param particles 粒子リスト
		// @param r_e 影響半径
		virtual Vector GetPressureGradient(const Particle::List& particles, const Grid& grid, const double& r_e, const double& dt, const double& rho, const double& n0) const
		{
			// 常に0
			Vector zero;
			zero[0] = 0;
			zero[1] = 0;
			return zero;
		}
	};
}
#endif