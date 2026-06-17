import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
# Ensure CPU platform if needed:
# jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.lax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.sem_grids import generate_tiled_sem_grid_2d
from geometry.jacobians import sem_jacobian
from operators.hybrid_ops import run_hybrid_sfx_2d_standard, run_hybrid_sfx_2d_sinc
from plots.diagnostics import plot_sfx_dashboard

# --- HELPERS ---
@jax.jit
def run_fv_proxy(u, cx, cy, dt, dx, dy):
    flux_x, flux_y = cx * u, cy * u
    d_flux_dx = (flux_x - jnp.roll(flux_x, 1, axis=0)) / dx
    d_flux_dy = (flux_y - jnp.roll(flux_y, 1, axis=1)) / dy
    return u - dt * (d_flux_dx + d_flux_dy)

def rk4_step(rhs_func, u, dt):
    k1 = rhs_func(u)
    k2 = rhs_func(u + 0.5 * dt * k1)
    k3 = rhs_func(u + 0.5 * dt * k2)
    k4 = rhs_func(u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Global Config
L, cx, cy, dt = 10.0, 1.0, 1.0, 0.05
total_time = 12.5
steps = int(total_time / dt)
N_fft, dx = 64, L/64

# Grids
X_fft, Y_fft = jnp.meshgrid(jnp.linspace(0, L, N_fft, endpoint=False), jnp.linspace(0, L, N_fft, endpoint=False))
ik_x = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[:, None]
ik_y = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[None, :]
u_init = jnp.exp(-((X_fft - 5)**2 + (Y_fft - 5)**2) / 1.5)

# SEM Setup
E_sem, P_sem = 32, 4
X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)
_, D_ref = get_sem_diff_matrix_2d(P_sem)
D_sem_global = jnp.kron(jnp.eye(E_sem), D_ref)
jac_sem = sem_jacobian(E_sem, L)
u_init_sem = jnp.exp(-((X_sem - 5)**2 + (Y_sem - 5)**2) / 1.5)

# Ribbon Configurations
# Standard: P=4 (5 nodes)
P_rib_std = 4
_, D_rib_std = get_sem_diff_matrix_2d(P_rib_std)
jac_rib_std = sem_jacobian(16, L)

# Optimized Sinc: P=2 (3 nodes)
P_rib_sinc = 1
_, D_rib_sinc = get_sem_diff_matrix_2d(P_rib_sinc)
jac_rib_sinc = sem_jacobian(16, L)

# --- SIMULATIONS ---

# 1. Pure FFT
t0 = time.time()
@jax.jit
def step_fft(u, _):
    return rk4_step(lambda u: -(cx * jnp.fft.ifft2(jnp.fft.fft2(u) * ik_x).real + cy * jnp.fft.ifft2(jnp.fft.fft2(u) * ik_y).real), u, dt), None
u_fft, _ = jax.lax.scan(step_fft, jnp.copy(u_init), None, length=steps)
t_fft = time.time() - t0

# 2. Pure SEM
t0 = time.time()
@jax.jit
def step_sem(u, _):
    return rk4_step(lambda u: -(cx * jnp.dot(D_sem_global, u) * jac_sem + cy * jnp.dot(u, D_sem_global.T) * jac_sem), u, dt), None
u_sem, _ = jax.lax.scan(step_sem, jnp.copy(u_init_sem), None, length=steps)
t_sem = time.time() - t0

# 3. Hybrid Standard (P=4)
t0 = time.time()
@jax.jit
def step_hyb_std(c, _):
    return run_hybrid_sfx_2d_standard(c[0], c[1], c[2], c[3], c[4], ik_x, ik_y, D_rib_std, jac_rib_std, cx, cy, dt), None
# Note: Initializing ribbons with shape specific to P_rib_std
init_state_std = (jnp.copy(u_init), u_init[:P_rib_std+1,:], u_init[-P_rib_std-1:,:], u_init[:,:P_rib_std+1], u_init[:,-P_rib_std-1:])
final_hyb_std, _ = jax.lax.scan(step_hyb_std, init_state_std, None, length=steps)
u_hyb_std = final_hyb_std[0]
t_hyb_std = time.time() - t0

# 4. Hybrid Sinc (Optimized P=2)
t0 = time.time()
@jax.jit
def step_hyb_sinc(c, _):
    return run_hybrid_sfx_2d_sinc(c[0], c[1], c[2], c[3], c[4], ik_x, ik_y, D_rib_sinc, jac_rib_sinc, cx, cy, dt), None
# Note: Initializing ribbons with shape specific to P_rib_sinc
init_state_sinc = (jnp.copy(u_init), u_init[:P_rib_sinc+1,:], u_init[-P_rib_sinc-1:,:], u_init[:,:P_rib_sinc+1], u_init[:,-P_rib_sinc-1:])
final_hyb_sinc, _ = jax.lax.scan(step_hyb_sinc, init_state_sinc, None, length=steps)
u_hyb_sinc = final_hyb_sinc[0]
t_hyb_sinc = time.time() - t0

# 5. FV Proxy
t0 = time.time()
@jax.jit
def step_fv(u, _):
    return run_fv_proxy(u, cx, cy, dt, dx, dx), None
u_fv, _ = jax.lax.scan(step_fv, jnp.copy(u_init), None, length=steps)
t_fv = time.time() - t0

# Post-processing
u_exact = jnp.exp(-(((X_fft - cx * total_time) % L - 5)**2 + ((Y_fft - cy * total_time) % L - 5)**2) / 1.5)
u_exact_sem = jnp.exp(-(((X_sem - cx * total_time) % L - 5)**2 + ((Y_sem - cy * total_time) % L - 5)**2) / 1.5)

err_fft = jnp.abs(u_fft - u_exact)
err_sem = jnp.abs(u_sem - u_exact_sem)
err_hyb_std = jnp.abs(u_hyb_std - u_exact)
err_hyb_sinc = jnp.abs(u_hyb_sinc - u_exact)
err_fv = jnp.abs(u_fv - u_exact)

# DoF Calculation
dof_fft = X_fft.size
dof_sem = X_sem.size
dof_hyb_std = dof_fft + (4 * N_fft * (P_rib_std + 1))
dof_hyb_sinc = dof_fft + (4 * N_fft * (P_rib_sinc + 1))

# --- REPORTING SECTION ---
print("\n" + "="*55)
print(f" SFX DCORE PERFORMANCE SUMMARY (t={total_time}s) ")
print("="*55)
def print_section(name, error, time, dof):
    print(f"[METHOD: {name}]")
    print(f" Max Error : {error:.2e}")
    print(f" Total Runtime : {time:.4f} s")
    print(f" Allocated Nodes : {dof:,} DoF")
    print("-" * 55)

print_section("Pure 2D FFT", jnp.max(err_fft), t_fft, dof_fft)
print_section("Pure 2D SEM", jnp.max(err_sem), t_sem, dof_sem)
print_section(f"Hybrid Standard (P={P_rib_std})", jnp.max(err_hyb_std), t_hyb_std, dof_hyb_std)
print_section(f"Hybrid Sinc (P={P_rib_sinc})", jnp.max(err_hyb_sinc), t_hyb_sinc, dof_hyb_sinc)
print_section("FV Proxy", jnp.max(err_fv), t_fv, dof_fft)
print("="*55)

plot_sfx_dashboard(
    L, X_fft, u_init, u_exact, u_fft, X_sem, u_sem, u_exact_sem, 
    u_hyb_std, u_hyb_sinc, u_fv, 
    err_fft, err_sem, err_hyb_std, err_hyb_sinc, err_fv, 
    t_fft, t_sem, t_hyb_std, t_hyb_sinc, t_fv
)

sys.exit(0)
