import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import jax
# Ensure CPU platform if needed:
#jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jax.lax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian
from operators.fft_ops import fft_rhs
from operators.hybrid_ops import run_hybrid_sfx_2d_standard, run_hybrid_sfx_2d_sinc
from operators.fv_ops import fv_rhs_1st_order, fv_rhs_3rd_order

from plots.diagnostics import plot_sfx_dashboard

# ============================================================
# TIMING UTILITIES
# ============================================================
def timed_run(name, fn):
    print(f"\nRunning {name}...")
    # compile + execute
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    compile_exec = time.perf_counter() - t0
    
    # execute only
    t0 = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    exec_only = time.perf_counter() - t0
    
    print(
        f"{name:<20}"
        f" compile+run={compile_exec:.6f}s "
        f"run={exec_only:.6f}s"
    )
    return result, exec_only

# ============================================================
# RK4
# ============================================================

def rk4_step(rhs_func, u, dt):

    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)

    return u + (dt / 6.0) * (
        k1 + 2 * k2 + 2 * k3 + k4
    )


# ============================================================
# FV PROXY
# ============================================================

@jax.jit
def step_fv(u, _):

    def rhs(q):
        return fv_rhs_3rd_order(
            q,
            cx,
            cy,
            dx,
            dx,
        )

    return rk4_step(rhs, u, dt), None

# ============================================================
# GLOBAL CONFIG
# ============================================================

L = 10.0

cx = 1.0
cy = 1.0

dt = 0.0001
total_time = 12.5

steps = int(total_time / dt)

N_fft = 1024 #128 #64
dx = L / N_fft

# ============================================================
# FFT GRID
# ============================================================

X_fft, Y_fft = jnp.meshgrid(
    jnp.linspace(0, L, N_fft, endpoint=False),
    jnp.linspace(0, L, N_fft, endpoint=False),
)

ik_x = (
    1j
    * 2
    * jnp.pi
    * jnp.fft.fftfreq(
        N_fft,
        d=L / N_fft,
    )
)[:, None]

ik_y = (
    1j
    * 2
    * jnp.pi
    * jnp.fft.fftfreq(
        N_fft,
        d=L / N_fft,
    )
)[None, :]

u_init = jnp.exp(
    -(
        (X_fft - 5) ** 2
        + (Y_fft - 5) ** 2
    )
    / 1.5
)

# ============================================================
# RIBBONS
# ============================================================

P_rib_std = 4
_, D_rib_std = get_sem_diff_matrix_2d(P_rib_std)
jac_rib_std = sem_jacobian(16, L)

P_rib_sinc = 4
_, D_rib_sinc = get_sem_diff_matrix_2d(P_rib_sinc)
jac_rib_sinc = sem_jacobian(16, L)

# --- SIMULATIONS ---
# ============================================================
# FFT SOLVER
# ============================================================

@jax.jit
def step_fft(u, _):
    # Now uses the modular fft_rhs and your existing rk4_step
    return rk4_step(lambda q: fft_rhs(q, ik_x, ik_y, cx, cy), u, dt), None

u_fft, t_fft = timed_run("PURE FFT", lambda: jax.lax.scan(step_fft, jnp.copy(u_init), None, length=steps)[0])

# ============================================================
# HYBRID STANDARD
# ============================================================

@jax.jit
def step_hyb_std(state, _):

    return (
        run_hybrid_sfx_2d_standard(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            ik_x,
            ik_y,
            D_rib_std,
            jac_rib_std,
            cx,
            cy,
            dt,
            dx
        ),
        None,
    )

init_state_std = (jnp.copy(u_init), u_init[:P_rib_std+1,:], u_init[-P_rib_std-1:,:], u_init[:,:P_rib_std+1], u_init[:,-P_rib_std-1:])
final_hyb_std, t_hyb_std = timed_run("HYBRID STD", lambda: jax.lax.scan(step_hyb_std, init_state_std, None, length=steps)[0])
u_hyb_std = final_hyb_std[0]

# ============================================================
# HYBRID SINC
# ============================================================

@jax.jit
def step_hyb_sinc(state, _):

    return (
        run_hybrid_sfx_2d_sinc(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            ik_x,
            ik_y,
            D_rib_sinc,
            jac_rib_sinc,
            cx,
            cy,
            dt,
            dx
        ),
        None,
    )

init_state_sinc = (jnp.copy(u_init), u_init[:P_rib_sinc+1,:], u_init[-P_rib_sinc-1:,:], u_init[:,:P_rib_sinc+1], u_init[:,-P_rib_sinc-1:])
final_hyb_sinc, t_hyb_sinc = timed_run("HYBRID SINC", lambda: jax.lax.scan(step_hyb_sinc, init_state_sinc, None, length=steps)[0])
u_hyb_sinc = final_hyb_sinc[0]

# ============================================================
# RUN BENCHMARKS
# ============================================================

u_fv, t_fv = timed_run(
    "FV RK4",
    lambda: jax.lax.scan(
        step_fv,
        jnp.copy(u_init),
        None,
        length=steps,
    )[0],
)

# Post-processing
u_exact = jnp.exp(-(((X_fft - cx * total_time) % L - 5)**2 + ((Y_fft - cy * total_time) % L - 5)**2) / 1.5)

err_fft = jnp.abs(u_fft - u_exact)
err_hyb_std = jnp.abs(u_hyb_std - u_exact)
err_hyb_sinc = jnp.abs(u_hyb_sinc - u_exact)
err_fv = jnp.abs(u_fv - u_exact)

dof_fft = X_fft.size
dof_hyb_std = dof_fft + (4 * N_fft * (P_rib_std + 1))
dof_hyb_sinc = dof_fft + (4 * N_fft * (P_rib_sinc + 1))

# --- REPORTING SECTION ---
print("\n" + "="*55)
print(f" SFX DCORE PERFORMANCE SUMMARY (t={total_time}s) ")
print("="*55)

def print_section(name, error, time, dof):
    print(f"[METHOD: {name}]")
    print(f" Max Error     : {error:.2e}")
    print(f" Total Runtime : {time:.4f} s")
    print(f" Allocated DoF : {dof:,}")
    print("-" * 55)
    
print_section("Pure 2D FFT", jnp.max(err_fft), t_fft, dof_fft)
print_section(f"Hybrid Standard (P={P_rib_std})", jnp.max(err_hyb_std), t_hyb_std, dof_hyb_std)
print_section(f"Hybrid Sinc (P={P_rib_sinc})", jnp.max(err_hyb_sinc), t_hyb_sinc, dof_hyb_sinc)
print_section("FV Proxy", jnp.max(err_fv), t_fv, dof_fft)

print("="*55)

plot_sfx_dashboard(L, X_fft, u_init, u_exact, u_fft, u_hyb_std, u_hyb_sinc, u_fv, err_fft, err_hyb_std, err_hyb_sinc, err_fv, t_fft, t_hyb_std, t_hyb_sinc, t_fv)
