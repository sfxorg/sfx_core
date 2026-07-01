import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
jax.config.update("jax_enable_x64", False)

import jax.numpy as jnp

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian

from operators.burgers_ops import (
    burgers_fft_rhs,
    burgers_fv_rhs_3rd_order,
    burgers_hybrid_standard,
)

# ============================================================
# TIMING
# ============================================================

def timed_run(name, fn):

    print(f"\nRunning {name}...")

    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    compile_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    run_only = time.perf_counter() - t0

    print(
        f"{name:<20}"
        f" compile+run={compile_run:.6f}s "
        f" run={run_only:.6f}s"
    )

    return result, run_only


# ============================================================
# RK4
# ============================================================

def rk4_step(rhs, u, dt):

    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)

    return u + (dt / 6.0) * (
        k1 +
        2.0 * k2 +
        2.0 * k3 +
        k4
    )


# ============================================================
# PROBLEM SETUP
# ============================================================

L = 10.0

N = 1024 #512 #256 #128

dx = L / N

dt = 1.0e-4

total_time = 0.5

steps = int(total_time / dt)

# ============================================================
# GRID
# ============================================================

X, Y = jnp.meshgrid(
    jnp.linspace(0.0, L, N, endpoint=False),
    jnp.linspace(0.0, L, N, endpoint=False),
    indexing="ij"
)

# ============================================================
# FOURIER WAVENUMBERS
# ============================================================

ik_x = (
    1j *
    2.0 *
    jnp.pi *
    jnp.fft.fftfreq(
        N,
        d=L / N
    )
)[:, None]

ik_y = (
    1j *
    2.0 *
    jnp.pi *
    jnp.fft.fftfreq(
        N,
        d=L / N
    )
)[None, :]

# ============================================================
# INITIAL CONDITION
# ============================================================

u_init = jnp.exp(
    -(
        (X - 5.0) ** 2
        +
        (Y - 5.0) ** 2
    ) / 1.5
)

# ============================================================
# SEM RIBBONS
# ============================================================

P = 8

_, D_rib = get_sem_diff_matrix_2d(P)

jac_rib = sem_jacobian(
    16,
    L
)

# ============================================================
# FFT SOLVER
# ============================================================

@jax.jit
def step_fft(u, _):

    return rk4_step(
        lambda q: burgers_fft_rhs(
            q,
            ik_x,
            ik_y
        ),
        u,
        dt
    ), None


u_fft, t_fft = timed_run(
    "BURGERS FFT",
    lambda: jax.lax.scan(
        step_fft,
        jnp.copy(u_init),
        None,
        length=steps
    )[0]
)

# ============================================================
# HYBRID
# ============================================================

@jax.jit
def step_hybrid(state, _):

    return (
        burgers_hybrid_standard(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            ik_x,
            ik_y,
            D_rib,
            jac_rib,
            dt,
            dx,
        ),
        None,
    )


init_hybrid = (
    jnp.copy(u_init),
    1.1*u_init[:P+1, :],
    1.1*u_init[-P-1:, :],
    1.1*u_init[:, :P+1],
    1.1*u_init[:, -P-1:]
)

final_hybrid, t_hybrid = timed_run(
    "BURGERS HYBRID",
    lambda: jax.lax.scan(
        step_hybrid,
        init_hybrid,
        None,
        length=steps
    )[0]
)

u_hybrid = final_hybrid[0]

fft_hyb_diff = jnp.max(
    jnp.abs(u_hybrid - u_fft)
)

print("\nACTIVE COUPLING DIAGNOSTIC")
print("--------------------------")
print(f"FFT-HYB Max Difference = {fft_hyb_diff:.6e}")

# ============================================================
# FV
# ============================================================

@jax.jit
def step_fv(u, _):

    def rhs(q):
        return burgers_fv_rhs_3rd_order(
            q,
            dx,
            dx
        )

    return rk4_step(
        rhs,
        u,
        dt
    ), None


u_fv, t_fv = timed_run(
    "BURGERS FV",
    lambda: jax.lax.scan(
        step_fv,
        jnp.copy(u_init),
        None,
        length=steps
    )[0]
)

# ============================================================
# ERRORS
# ============================================================

err_hybrid = jnp.abs(
    u_hybrid - u_fft
)

err_fv = jnp.abs(
    u_fv - u_fft
)

# ============================================================
# REPORT
# ============================================================

dof_fft = N * N
dof_hybrid = dof_fft + (
    4 * N * (P + 1)
)

print()
print("=" * 60)
print(" 2D BURGERS PERFORMANCE SUMMARY")
print("=" * 60)

print("[FFT]")
print(f" Runtime      : {t_fft:.4f} s")
print(f" DoF          : {dof_fft:,}")
print("-" * 60)

print("[HYBRID]")
print(f" Runtime      : {t_hybrid:.4f} s")
print(f" Max Error    : {jnp.max(err_hybrid):.4e}")
print(f" L2 Error     : {jnp.sqrt(jnp.mean(err_hybrid**2)):.4e}")
print(f" DoF          : {dof_hybrid:,}")
print("-" * 60)

print("[FV]")
print(f" Runtime      : {t_fv:.4f} s")
print(f" Max Error    : {jnp.max(err_fv):.4e}")
print(f" L2 Error     : {jnp.sqrt(jnp.mean(err_fv**2)):.4e}")
print(f" DoF          : {dof_fft:,}")
print("-" * 60)

print("=" * 60)

print(
    f"FV speedup vs FFT : {t_fft/t_fv:.2f}x"
)

print(
    f"HYB speedup vs FFT : {t_fft/t_hybrid:.2f}x"
)

print(jnp.max(jnp.abs(u_fft)))
print(jnp.min(u_fft))
print(
    "top mismatch:",
    float(
        jnp.max(
            jnp.abs(
                final_hybrid[1][0:2,:]
                - u_hybrid[-2:,:]
            )
        )
    )
)

print(
    "top mismatch",
    jnp.max(
        jnp.abs(
            final_hybrid[1][0:2,:]
            - u_hybrid[-2:,:]
        )
    )
)

print(
    "bot mismatch",
    jnp.max(
        jnp.abs(
            final_hybrid[2][-2:,:]
            - u_hybrid[0:2,:]
        )
    )
)

print(
    "left mismatch",
    jnp.max(
        jnp.abs(
            final_hybrid[3][:,-2:]
            - u_hybrid[:,0:2]
        )
    )
)

print(
    "right mismatch",
    jnp.max(
        jnp.abs(
            final_hybrid[4][:,0:2]
            - u_hybrid[:,-2:]
        )
    )
)

print(
    "{:.16e}".format(
        float(
            jnp.max(
                jnp.abs(u_hybrid-u_fft)
            )
        )
    )
)

energy_fft = jnp.mean(u_fft**2)
energy_hyb = jnp.mean(u_hybrid**2)

print(
    "energy diff",
    float(abs(energy_fft-energy_hyb))
)

mass_fft = jnp.mean(u_fft)
mass_hyb = jnp.mean(u_hybrid)

print(
    "mass diff =",
    float(abs(mass_fft - mass_hyb))
)

print(
    "max =", float(jnp.max(u_hybrid))
)

print(
    "min =", float(jnp.min(u_hybrid))
)

# ============================================================
# RIBBON / INTERFACE DIAGNOSTICS
# ============================================================

E_fft = jnp.mean(u_hybrid**2)

E_rib_top   = jnp.mean(final_hybrid[1]**2)
E_rib_bot   = jnp.mean(final_hybrid[2]**2)
E_rib_left  = jnp.mean(final_hybrid[3]**2)
E_rib_right = jnp.mean(final_hybrid[4]**2)

E_rib_total = (
    E_rib_top +
    E_rib_bot +
    E_rib_left +
    E_rib_right
)

top_rms = jnp.sqrt(
    jnp.mean(
        (final_hybrid[1][0:2, :]
         - u_hybrid[-2:, :])**2
    )
)

bot_rms = jnp.sqrt(
    jnp.mean(
        (final_hybrid[2][-2:, :]
         - u_hybrid[0:2, :])**2
    )
)

left_rms = jnp.sqrt(
    jnp.mean(
        (final_hybrid[3][:, -2:]
         - u_hybrid[:, 0:2])**2
    )
)

right_rms = jnp.sqrt(
    jnp.mean(
        (final_hybrid[4][:, 0:2]
         - u_hybrid[:, -2:])**2
    )
)

print("\nRIBBON / INTERFACE DIAGNOSTICS")
print("--------------------------------")
print("FFT energy           =", float(E_fft))
print("Ribbon energy total  =", float(E_rib_total))
print(
    "Ribbon/FFT (%)       =",
    float(100.0 * E_rib_total / E_fft)
)
print("Top ribbon energy    =", float(E_rib_top))
print("Bot ribbon energy    =", float(E_rib_bot))
print("Left ribbon energy   =", float(E_rib_left))
print("Right ribbon energy  =", float(E_rib_right))
print("Top RMS mismatch     =", float(top_rms))
print("Bot RMS mismatch     =", float(bot_rms))
print("Left RMS mismatch    =", float(left_rms))
print("Right RMS mismatch   =", float(right_rms))
print(
    "Ribbon max amplitude =",
    float(
        jnp.max(
            jnp.abs(
                jnp.concatenate([
                    final_hybrid[1].ravel(),
                    final_hybrid[2].ravel(),
                    final_hybrid[3].ravel(),
                    final_hybrid[4].ravel(),
                ])
            )
        )
    )
)
print(
    "Ribbon/FFT max ratio =",
    float(
        jnp.max(
            jnp.abs(
                jnp.concatenate([
                    final_hybrid[1].ravel(),
                    final_hybrid[2].ravel(),
                    final_hybrid[3].ravel(),
                    final_hybrid[4].ravel(),
                ])
            )
        )
        /
        jnp.max(jnp.abs(u_hybrid))
    )
)

print()
print("="*60)
print("INTERFACE SPECTRAL COMPRESSIBILITY")
print("="*60)

def analyze_edge(name, edge_err):

    signal = jnp.mean(edge_err, axis=0)

    spec = jnp.fft.rfft(signal)

    e = jnp.abs(spec)**2

    e = e / (jnp.sum(e) + 1e-30)

    cume = jnp.cumsum(e)

    k95 = int(jnp.argmax(cume > 0.95))
    k99 = int(jnp.argmax(cume > 0.99))

    print()
    print(name)
    print("RMS       =", float(jnp.sqrt(jnp.mean(edge_err**2))))
    print("Mode<=4   =", float(cume[min(4, len(cume)-1)]))
    print("Mode<=8   =", float(cume[min(8, len(cume)-1)]))
    print("Mode<=16  =", float(cume[min(16, len(cume)-1)]))
    print("Mode<=32  =", float(cume[min(32, len(cume)-1)]))
    print("95% mode  =", k95)
    print("99% mode  =", k99)

top_err = (
    final_hybrid[1][0:2,:]
    - u_hybrid[-2:,:]
)

bot_err = (
    final_hybrid[2][-2:,:]
    - u_hybrid[0:2,:]
)

left_err = (
    final_hybrid[3][:,-2:]
    - u_hybrid[:,0:2]
)

right_err = (
    final_hybrid[4][:,0:2]
    - u_hybrid[:,-2:]
)

analyze_edge("TOP", top_err)
analyze_edge("BOTTOM", bot_err)
analyze_edge("LEFT", left_err.T)
analyze_edge("RIGHT", right_err.T)
