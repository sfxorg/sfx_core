import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
# Force CPU for environment stability
jax.config.update("jax_platform_name", "cpu")
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
L, dt = 10.0, 0.005 # Using smaller dt for high-velocity stability
total_time = 2.0 
steps = int(total_time / dt)
N_fft, dx = 64, L/64

# Grids
X_fft, Y_fft = jnp.meshgrid(jnp.linspace(0, L, N_fft, endpoint=False), jnp.linspace(0, L, N_fft, endpoint=False))
ik_x = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[:, None]
ik_y = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[None, :]
u_init = jnp.exp(-((X_fft - 2)**2 + (Y_fft - 2)**2) / 1.5)

# SEM Setup
E_sem, P_sem = 32, 4
X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)
_, D_ref = get_sem_diff_matrix_2d(P_sem)
D_sem_global = jnp.kron(jnp.eye(E_sem), D_ref)
jac_sem = sem_jacobian(E_sem, L)
u_init_sem = jnp.exp(-((X_sem - 2)**2 + (Y_sem - 2)**2) / 1.5)

# Hybrid Setup
P_rib_std, P_rib_sinc = 4, 1
_, D_rib_std = get_sem_diff_matrix_2d(P_rib_std)
_, D_rib_sinc = get_sem_diff_matrix_2d(P_rib_sinc)
jac_rib_std = sem_jacobian(16, L)
jac_rib_sinc = sem_jacobian(16, L)

# --- VELOCITY SWEEP ---
velocities = [1.0, 5.0, 10.0]
print(f"\n{'Velocity':<10} | {'Pure FFT':<12} | {'Hybrid Sinc':<12} | {'FV Proxy':<12}")
print("-" * 55)

for vel in velocities:
    cx, cy = vel, vel

    # 1. FFT
    t0 = time.time()
    @jax.jit
    def step_fft(u, _):
        return rk4_step(lambda u: -(cx * jnp.fft.ifft2(jnp.fft.fft2(u) * ik_x).real + cy * jnp.fft.ifft2(jnp.fft.fft2(u) * ik_y).real), u, dt), None
    u_fft, _ = jax.lax.scan(step_fft, jnp.copy(u_init), None, length=steps)
    t_fft = time.time() - t0

    # 2. SEM
    t0 = time.time()
    @jax.jit
    def step_sem(u, _):
        return rk4_step(lambda u: -(cx * jnp.dot(D_sem_global, u) * jac_sem + cy * jnp.dot(u, D_sem_global.T) * jac_sem), u, dt), None
    u_sem, _ = jax.lax.scan(step_sem, jnp.copy(u_init_sem), None, length=steps)
    t_sem = time.time() - t0

    # 3. Hybrid Standard
    t0 = time.time()
    @jax.jit
    def step_hyb_std(c, _):
        return run_hybrid_sfx_2d_standard(c[0], c[1], c[2], c[3], c[4], ik_x, ik_y, D_rib_std, jac_rib_std, cx, cy, dt), None
    final_hyb_std, _ = jax.lax.scan(step_hyb_std, (jnp.copy(u_init), u_init[:P_rib_std+1,:], u_init[-P_rib_std-1:,:], u_init[:,:P_rib_std+1], u_init[:,-P_rib_std-1:]), None, length=steps)
    u_hyb_std = final_hyb_std[0]
    t_hyb_std = time.time() - t0

    # 4. Hybrid Sinc
    t0 = time.time()
    @jax.jit
    def step_hyb_sinc(c, _):
        return run_hybrid_sfx_2d_sinc(c[0], c[1], c[2], c[3], c[4], ik_x, ik_y, D_rib_sinc, jac_rib_sinc, cx, cy, dt), None
    final_hyb_sinc, _ = jax.lax.scan(step_hyb_sinc, (jnp.copy(u_init), u_init[:P_rib_sinc+1,:], u_init[-P_rib_sinc-1:,:], u_init[:,:P_rib_sinc+1], u_init[:,-P_rib_sinc-1:]), None, length=steps)
    u_hyb_sinc = final_hyb_sinc[0]
    t_hyb_sinc = time.time() - t0

    # 5. FV Proxy
    t0 = time.time()
    @jax.jit
    def step_fv(u, _):
        return run_fv_proxy(u, cx, cy, dt, dx, dx), None
    u_fv, _ = jax.lax.scan(step_fv, jnp.copy(u_init), None, length=steps)
    t_fv = time.time() - t0

    # Errors
    u_exact = jnp.exp(-((((X_fft - cx * total_time) % L) - 2)**2 + (((Y_fft - cy * total_time) % L) - 2)**2) / 1.5)
    u_exact_sem = jnp.exp(-((((X_sem - cx * total_time) % L) - 2)**2 + (((Y_sem - cy * total_time) % L) - 2)**2) / 1.5)
    
    err_fft = jnp.max(jnp.abs(u_fft - u_exact))
    err_sinc = jnp.max(jnp.abs(u_hyb_sinc - u_exact))
    err_fv = jnp.max(jnp.abs(u_fv - u_exact))

    print(f"{vel:<10.1f} | {err_fft:<12.2e} | {err_sinc:<12.2e} | {err_fv:<12.2e}")

    # History for dashboard (using last velocity)
    results_history = (u_init, u_exact, u_fft, u_sem, u_exact_sem, u_hyb_std, u_hyb_sinc, u_fv, 
                       jnp.abs(u_fft - u_exact), jnp.abs(u_sem - u_exact_sem), jnp.abs(u_hyb_std - u_exact), 
                       jnp.abs(u_hyb_sinc - u_exact), jnp.abs(u_fv - u_exact), t_fft, t_sem, t_hyb_std, t_hyb_sinc, t_fv)

# --- PLOTTING ---
plot_sfx_dashboard(
    L, X_fft, results_history[0], results_history[1], results_history[2], 
    X_sem, results_history[3], results_history[4], results_history[5], 
    results_history[6], results_history[7], results_history[8], 
    results_history[9], results_history[10], results_history[11], 
    results_history[12], results_history[13], results_history[14], 
    results_history[15], results_history[16], results_history[17]
)

print("\nSTRESS TEST COMPLETE. Dashboard saved.")
sys.exit(0)
