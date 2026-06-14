import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import jax
import jax.numpy as jnp
import jax.lax
import matplotlib.pyplot as plt

# CRITICAL: Precision and Stability
jax.config.update("jax_enable_x64", True)

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.sem_grids import generate_tiled_sem_grid_2d
from geometry.jacobians import sem_jacobian
from operators.fft_ops import run_pure_fft_2d
from operators.sem_ops import run_pure_sem_2d
from operators.hybrid_ops import run_hybrid_sfx_2d_stable
from plots.diagnostics import plot_sfx_dashboard

# RK4 Helper
def rk4_step(rhs_func, u, dt):
    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Global Config
L = 10.0
cx, cy = 1.0, 1.0
dt = 0.0005
total_time = 12.5
steps = int(total_time / dt)

# FFT Grid
N_fft = 64
dx_fft = L / N_fft
X_fft, Y_fft = jnp.meshgrid(jnp.linspace(0, L, N_fft, endpoint=False), jnp.linspace(0, L, N_fft, endpoint=False))
ik_x = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=dx_fft)[:, None]
ik_y = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=dx_fft)[None, :]
u_init_fft = jnp.exp(-((X_fft - 5)**2 + (Y_fft - 5)**2) / 1.5)

# SEM Grid
E_sem, P_sem = 32, 4
X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)
_, D_ref = get_sem_diff_matrix_2d(P_sem)
D_sem_global = jnp.kron(jnp.eye(E_sem), D_ref)
jac_sem = sem_jacobian(E_sem, L)
u_init_sem = jnp.exp(-((X_sem - 5)**2 + (Y_sem - 5)**2) / 1.5)

# Hybrid Ribbons
P_rib = 4
_, D_rib = get_sem_diff_matrix_2d(P_rib)
jac_rib = sem_jacobian(16, L)
u_rib_bot = u_init_fft[:P_rib+1, :]
u_rib_top = u_init_fft[-P_rib-1:, :]
u_rib_left = u_init_fft[:, :P_rib+1]
u_rib_right = u_init_fft[:, -P_rib-1:]

# Scan Setup
@jax.jit
def step_fft(u, _):
    def rhs(u):
        u_hat = jnp.fft.fft2(u)
        return -(cx * jnp.fft.ifft2(u_hat * ik_x).real + cy * jnp.fft.ifft2(u_hat * ik_y).real)
    return rk4_step(rhs, u, dt), None

@jax.jit
def step_sem(u, _):
    def rhs(u):
        return -(cx * jnp.dot(D_sem_global, u) * jac_sem + cy * jnp.dot(u, D_sem_global.T) * jac_sem)
    return rk4_step(rhs, u, dt), None

@jax.jit
def step_hyb(carry, _):
    return run_hybrid_sfx_2d_stable(carry[0], carry[1], carry[2], carry[3], carry[4], ik_x, ik_y, D_rib, jac_rib, cx, cy, dt), None

# Execution
t0 = time.time()
u_fft, _ = jax.lax.scan(step_fft, jnp.copy(u_init_fft), None, length=steps)
t_fft = time.time() - t0

t0 = time.time()
u_sem, _ = jax.lax.scan(step_sem, jnp.copy(u_init_sem), None, length=steps)
t_sem = time.time() - t0

t0 = time.time()
initial_state = (jnp.copy(u_init_fft), u_rib_top, u_rib_bot, u_rib_left, u_rib_right)
final_state, _ = jax.lax.scan(step_hyb, initial_state, None, length=steps)
u_hyb_fft = final_state[0]
t_hyb = time.time() - t0

# Reporting (Kept as requested)
X_shift = (X_fft - cx * total_time) % L
Y_shift = (Y_fft - cy * total_time) % L
u_exact = jnp.exp(-((X_shift - 5)**2 + (Y_shift - 5)**2) / 1.5)
X_sem_shift = (X_sem - cx * total_time) % L
Y_sem_shift = (Y_sem - cy * total_time) % L
u_exact_sem = jnp.exp(-((X_sem_shift - 5)**2 + (Y_sem_shift - 5)**2) / 1.5)

err_fft = jnp.abs(u_fft - u_exact)
err_sem = jnp.abs(u_sem - u_exact_sem)
err_hyb = jnp.abs(u_hyb_fft - u_exact)

print("\n" + "="*55)
print(f" 2D REBALANCED CORE ACCURACY REPORT (t={total_time}s) ")
print("="*55)
print(f"Case 1: Pure 2D FFT Max Error : {jnp.max(err_fft):.2e}")
print(f"Case 2: Pure 2D SEM Max Error : {jnp.max(err_sem):.2e}")
print(f"Case 3: Inverted Hybrid Max Error : {jnp.max(err_hyb):.2e}")
print("-"*55)
print(f"Case 1 Baseline Run Time : {t_fft:.4f} seconds")
print(f"Case 2 Tiled Mesh Run Time : {t_sem:.4f} seconds")
print(f"Case 3 Hybrid Engine Run Time : {t_hyb:.4f} seconds")
print("-"*55)
print(f"Case 1 Total Allocated Grid Nodes : {X_fft.size:,} DoF")
print(f"Case 2 Total Allocated Grid Nodes : {X_sem.size:,} DoF")
print(f"Case 3 Total Allocated Grid Nodes : {u_hyb_fft.size + u_rib_bot.size + u_rib_top.size + u_rib_left.size + u_rib_right.size:,} DoF")
print("="*55)
plot_sfx_dashboard( L, X_fft, u_exact, u_fft, X_sem, u_sem, u_exact_sem, u_hyb_fft, err_fft, err_sem, err_hyb, t_fft, t_sem, t_hyb )
