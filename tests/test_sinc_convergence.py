import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian
from operators.hybrid_ops import run_hybrid_sfx_2d_sinc

# ============================================================
# CONFIG
# ============================================================

L = 10.0
N_fft = 64

cx = 1.0
cy = 1.0

dt = 0.05
total_time = 12.5
steps = int(total_time / dt)

print("\n" + "=" * 70)
print("SINC RIBBON CONVERGENCE STUDY")
print("=" * 70)

# ============================================================
# FFT GRID
# ============================================================

X_fft, Y_fft = jnp.meshgrid(
    jnp.linspace(0, L, N_fft, endpoint=False),
    jnp.linspace(0, L, N_fft, endpoint=False)
)

ik_x = (
    1j
    * 2.0
    * jnp.pi
    * jnp.fft.fftfreq(N_fft, d=L / N_fft)
)[:, None]

ik_y = (
    1j
    * 2.0
    * jnp.pi
    * jnp.fft.fftfreq(N_fft, d=L / N_fft)
)[None, :]

u_init = jnp.exp(
    -((X_fft - 5.0) ** 2 + (Y_fft - 5.0) ** 2) / 1.5
)

# ============================================================
# PURE FFT REFERENCE
# ============================================================

def rk4_step(rhs_func, u, dt):
    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)

    return u + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )

@jax.jit
def step_fft(u, _):

    def rhs(v):

        vhat = jnp.fft.fft2(v)

        dudx = jnp.fft.ifft2(vhat * ik_x).real
        dudy = jnp.fft.ifft2(vhat * ik_y).real

        return -(cx * dudx + cy * dudy)

    return rk4_step(rhs, u, dt), None

print("\nComputing FFT reference...")

u_fft_ref, _ = jax.lax.scan(
    step_fft,
    jnp.copy(u_init),
    None,
    length=steps,
)

jax.block_until_ready(u_fft_ref)

# ============================================================
# EXACT SOLUTION
# ============================================================

u_exact = jnp.exp(
    -(
        (((X_fft - cx * total_time) % L) - 5.0) ** 2
        +
        (((Y_fft - cy * total_time) % L) - 5.0) ** 2
    )
    / 1.5
)

fft_error = float(
    jnp.max(jnp.abs(u_fft_ref - u_exact))
)

print(f"\nFFT Reference Error = {fft_error:.6e}")

# ============================================================
# SWEEP P = 1..4
# ============================================================

results = []

for P_rib in [1, 2, 3, 4]:

    print("\n" + "-" * 70)
    print(f"Running P_rib = {P_rib}")
    print("-" * 70)

    _, D_rib = get_sem_diff_matrix_2d(P_rib)

    jac_rib = sem_jacobian(16, L)

    @jax.jit
    def step_sinc(carry, _):

        return (
            run_hybrid_sfx_2d_sinc(
                carry[0],
                carry[1],
                carry[2],
                carry[3],
                carry[4],
                ik_x,
                ik_y,
                D_rib,
                jac_rib,
                cx,
                cy,
                dt,
            ),
            None,
        )

    init_state = (
        jnp.copy(u_init),
        u_init[:P_rib + 1, :],
        u_init[-P_rib - 1:, :],
        u_init[:, :P_rib + 1],
        u_init[:, -P_rib - 1:],
    )

    t0 = time.time()

    final_state, _ = jax.lax.scan(
        step_sinc,
        init_state,
        None,
        length=steps,
    )

    u_fft_component = final_state[0]

    jax.block_until_ready(u_fft_component)

    runtime = time.time() - t0

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    error = float(
        jnp.max(
            jnp.abs(
                u_fft_component - u_exact
            )
        )
    )

    # --------------------------------------------------------
    # MATCH TO FFT REFERENCE
    # --------------------------------------------------------

    difference = float(
        jnp.max(
            jnp.abs(
                u_fft_component - u_fft_ref
            )
        )
    )

    # --------------------------------------------------------
    # RIBBON DOF
    # --------------------------------------------------------

    ribbon_dof = (
        final_state[1].size +
        final_state[2].size +
        final_state[3].size +
        final_state[4].size
    )

    mass = float(jnp.sum(u_fft_component))

    print(f"P                 : {P_rib}")
    print(f"Ribbon DoF        : {ribbon_dof}")
    print(f"Runtime (s)       : {runtime:.4f}")
    print(f"Solution Error    : {error:.6e}")
    print(f"FFT Difference    : {difference:.6e}")
    print(f"Mass              : {mass:.12f}")

    results.append(
        (
            P_rib,
            ribbon_dof,
            runtime,
            error,
            difference,
            mass,
        )
    )

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 110)
print("SUMMARY")
print("=" * 110)

header = (
    f"{'P':>4} "
    f"{'RibbonDoF':>12} "
    f"{'Runtime(s)':>12} "
    f"{'Error':>14} "
    f"{'FFTDiff':>14}"
)

print(header)
print("-" * len(header))

for r in results:

    p = r[0]
    dof = r[1]
    rt = r[2]
    err = r[3]
    diff = r[4]

    print(
        f"{p:>4d} "
        f"{dof:>12d} "
        f"{rt:>12.4f} "
        f"{err:>14.6e} "
        f"{diff:>14.6e}"
    )

print("\nDone.")
