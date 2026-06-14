import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import jax
import jax.numpy as jnp
import jax.lax
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
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

# --- AI SURROGATE HELPER ---
def ai_surrogate_model(u_true, cutoff=0.5):
    u_hat = jnp.fft.fft2(u_true)
    N = u_true.shape[0]
    freqs = jnp.fft.fftfreq(N)
    kx, ky = jnp.meshgrid(freqs, freqs)
    dist = jnp.sqrt(kx**2 + ky**2)
    mask = jnp.exp(-dist**2 / cutoff**2)
    return jnp.fft.ifft2(u_hat * mask).real

# RK4 Helper
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

# Grids
N_fft = 64
X_fft, Y_fft = jnp.meshgrid(jnp.linspace(0, L, N_fft, endpoint=False), jnp.linspace(0, L, N_fft, endpoint=False))
ik_x = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[:, None]
ik_y = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[None, :]
u_init_fft = jnp.exp(-((X_fft - 5)**2 + (Y_fft - 5)**2) / 1.5)

E_sem, P_sem = 32, 4
X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)
_, D_ref = get_sem_diff_matrix_2d(P_sem)
D_sem_global = jnp.kron(jnp.eye(E_sem), D_ref)
jac_sem = sem_jacobian(E_sem, L)
u_init_sem = jnp.exp(-((X_sem - 5)**2 + (Y_sem - 5)**2) / 1.5)

P_rib = 4
_, D_rib = get_sem_diff_matrix_2d(P_rib)
jac_rib = sem_jacobian(16, L)
u_rib_bot = u_init_fft[:P_rib+1, :]
u_rib_top = u_init_fft[-P_rib-1:, :]
u_rib_left = u_init_fft[:, :P_rib+1]
u_rib_right = u_init_fft[:, -P_rib-1:]

# Run Simulations
t0 = time.time()
@jax.jit
def step_fft(u, _): return rk4_step(lambda u: -(cx * jnp.fft.ifft2(jnp.fft.fft2(u) * ik_x).real + cy * jnp.fft.ifft2(jnp.fft.fft2(u) * ik_y).real), u, dt), None
u_fft, _ = jax.lax.scan(step_fft, jnp.copy(u_init_fft), None, length=steps)
t_fft = time.time() - t0

t0 = time.time()
@jax.jit
def step_sem(u, _): return rk4_step(lambda u: -(cx * jnp.dot(D_sem_global, u) * jac_sem + cy * jnp.dot(u, D_sem_global.T) * jac_sem), u, dt), None
u_sem, _ = jax.lax.scan(step_sem, jnp.copy(u_init_sem), None, length=steps)
t_sem = time.time() - t0

t0 = time.time()
@jax.jit
def step_hyb(carry, _): return run_hybrid_sfx_2d_stable(carry[0], carry[1], carry[2], carry[3], carry[4], ik_x, ik_y, D_rib, jac_rib, cx, cy, dt), None
final_state, _ = jax.lax.scan(step_hyb, (jnp.copy(u_init_fft), u_rib_top, u_rib_bot, u_rib_left, u_rib_right), None, length=steps)
u_hyb_fft = final_state[0]
t_hyb = time.time() - t0

# Calculations
X_shift, Y_shift = (X_fft - cx * total_time) % L, (Y_fft - cy * total_time) % L
u_exact = jnp.exp(-((X_shift - 5)**2 + (Y_shift - 5)**2) / 1.5)
X_sem_shift, Y_sem_shift = (X_sem - cx * total_time) % L, (Y_sem - cy * total_time) % L
u_exact_sem = jnp.exp(-((X_sem_shift - 5)**2 + (Y_sem_shift - 5)**2) / 1.5)
u_ai = ai_surrogate_model(u_exact)

err_fft, err_sem, err_hyb, err_ai = jnp.abs(u_fft-u_exact), jnp.abs(u_sem-u_exact_sem), jnp.abs(u_hyb_fft-u_exact), jnp.abs(u_ai-u_exact)

# Reporting Section
print("\n" + "="*55)
print(f" SFX DCORE PERFORMANCE SUMMARY (t={total_time}s) ")
print("="*55)

def print_section(name, error, time, dof):
    print(f"[METHOD: {name}]")
    print(f"  Max Error       : {error:.2e}")
    print(f"  Run Time        : {time:.4f} s")
    print(f"  Allocated Nodes : {dof:,} DoF")
    print("-"*55)

print_section("Pure 2D FFT", jnp.max(err_fft), t_fft, X_fft.size)
print_section("Pure 2D SEM", jnp.max(err_sem), t_sem, X_sem.size)
print_section("Hybrid Engine", jnp.max(err_hyb), t_hyb, u_hyb_fft.size + u_rib_bot.size + u_rib_top.size + u_rib_left.size + u_rib_right.size)
print_section("AI Surrogate", jnp.max(err_ai), 0.0, X_fft.size)
print("="*55)

# --- AI SENSITIVITY ANALYSIS ---
print("\n" + "="*55)
print(" AI SURROGATE SENSITIVITY ANALYSIS ")
print("="*55)
# Lower cutoff = more blur (worse AI), Higher cutoff = less blur (better AI)
test_cutoffs = [0.2, 0.4, 0.6, 0.8, 1.0]

best_ai_u = None
best_ai_err = None

for c in test_cutoffs:
    u_ai_test = ai_surrogate_model(u_exact, cutoff=c)
    err_ai_test = jnp.max(jnp.abs(u_ai_test - u_exact))
    print(f"AI Cutoff {c:.1f}: Max Error = {err_ai_test:.2e}")
    
    # Save the 'best' one (e.g., cutoff 0.8) for the final dashboard
    if c == 0.8:
        best_ai_u = u_ai_test
        best_ai_err = jnp.abs(u_ai_test - u_exact)

# Use the 'best' AI result for the plotting call
u_ai = best_ai_u
err_ai = best_ai_err
print("="*55 + "\n")

# Plotting call
plot_sfx_dashboard(L, X_fft, u_exact, u_fft, X_sem, u_sem, u_exact_sem, 
                   u_hyb_fft, u_ai, err_fft, err_sem, err_hyb, err_ai, 
                   t_fft, t_sem, t_hyb)

sys.exit(0)
