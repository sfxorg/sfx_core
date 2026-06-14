import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax.numpy as jnp
import matplotlib.pyplot as plt

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.sem_grids import generate_tiled_sem_grid_2d
from geometry.jacobians import sem_jacobian

from operators.fft_ops import run_pure_fft_2d
from operators.sem_ops import run_pure_sem_2d
from operators.hybrid_ops import run_hybrid_sfx_2d_stable
from plots.diagnostics import plot_sfx_dashboard

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================
L = 10.0
cx, cy = 1.0, 1.0
#dt = 0.00005
dt = 0.0005
total_time = 12.5
steps = int(total_time / dt)


# ============================================================
# 2. FFT GRID
# ============================================================
N_fft = 64
dx_fft = L / N_fft

X_fft, Y_fft = jnp.meshgrid(
    jnp.linspace(0, L, N_fft, endpoint=False),
    jnp.linspace(0, L, N_fft, endpoint=False)
)

ik_x = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=dx_fft)[:, None]
ik_y = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=dx_fft)[None, :]

u_init_fft = jnp.exp(-((X_fft - 5)**2 + (Y_fft - 5)**2) / 1.5)


# ============================================================
# 3. SEM GRID
# ============================================================
E_sem, P_sem = 32, 4
X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)

_, D_ref = get_sem_diff_matrix_2d(P_sem)
D_sem_global = jnp.kron(jnp.eye(E_sem), D_ref)
jac_sem = sem_jacobian(E_sem, L)

u_init_sem = jnp.exp(-((X_sem - 5)**2 + (Y_sem - 5)**2) / 1.5)


# ============================================================
# 4. HYBRID RIBBONS
# ============================================================
P_rib = 4
_, D_rib = get_sem_diff_matrix_2d(P_rib)
jac_rib = sem_jacobian(16, L)

u_hyb_fft = jnp.copy(u_init_fft)
u_rib_bot = u_init_fft[:P_rib+1, :]
u_rib_top = u_init_fft[-P_rib-1:, :]
u_rib_left = u_init_fft[:, :P_rib+1]
u_rib_right = u_init_fft[:, -P_rib-1:]


# ============================================================
# 5. RUN SIMULATIONS
# ============================================================
print(f"Running Upgraded Matched 2D Framework for {steps} steps until t={total_time}s...")

# FFT
t0 = time.time()
u_fft = jnp.copy(u_init_fft)
for _ in range(steps):
    u_fft = run_pure_fft_2d(u_fft, ik_x, ik_y, cx, cy, dt)
u_fft.block_until_ready()
t_fft = time.time() - t0

# SEM
t0 = time.time()
u_sem = jnp.copy(u_init_sem)
for _ in range(steps):
    u_sem = run_pure_sem_2d(u_sem, D_sem_global, jac_sem, cx, cy, dt)
u_sem.block_until_ready()
t_sem = time.time() - t0

# HYBRID
t0 = time.time()
for _ in range(steps):
    u_hyb_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right = run_hybrid_sfx_2d_stable(
        u_hyb_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right,
        ik_x, ik_y, D_rib, jac_rib, cx, cy, dt
    )
u_hyb_fft.block_until_ready()
t_hyb = time.time() - t0


# ============================================================
# 6. EXACT SOLUTION
# ============================================================
X_shift = (X_fft - cx * total_time) % L
Y_shift = (Y_fft - cy * total_time) % L
u_exact = jnp.exp(-((X_shift - 5)**2 + (Y_shift - 5)**2) / 1.5)

X_sem_shift = (X_sem - cx * total_time) % L
Y_sem_shift = (Y_sem - cy * total_time) % L
u_exact_sem = jnp.exp(-((X_sem_shift - 5)**2 + (Y_sem_shift - 5)**2) / 1.5)


# ============================================================
# 7. ERRORS
# ============================================================
err_fft = jnp.abs(u_fft - u_exact)
err_sem = jnp.abs(u_sem - u_exact_sem)
err_hyb = jnp.abs(u_hyb_fft - u_exact)


# ============================================================
# 8. REPORT
# ============================================================
print("\n" + "="*55)
print(f"       2D REBALANCED CORE ACCURACY REPORT (t={total_time}s) ")
print("="*55)
print(f"Case 1: Pure 2D FFT Max Error        : {jnp.max(err_fft):.2e}")
print(f"Case 2: Pure 2D SEM Max Error        : {jnp.max(err_sem):.2e}")
print(f"Case 3: Inverted Hybrid Max Error    : {jnp.max(err_hyb):.2e}")
print("-"*55)
print(f"Case 1 Baseline Run Time             : {t_fft:.4f} seconds")
print(f"Case 2 Tiled Mesh Run Time           : {t_sem:.4f} seconds")
print(f"Case 3 Hybrid Engine Run Time        : {t_hyb:.4f} seconds")
print("-"*55)
print(f"Case 1 Total Allocated Grid Nodes    : {X_fft.size:,} DoF")
print(f"Case 2 Total Allocated Grid Nodes    : {X_sem.size:,} DoF")
print(f"Case 3 Total Allocated Grid Nodes    : {u_hyb_fft.size + u_rib_bot.size + u_rib_top.size + u_rib_left.size + u_rib_right.size:,} DoF")
print("="*55)

plot_sfx_dashboard(
    L,
    X_fft, u_exact, u_fft,
    X_sem, u_sem, u_exact_sem,
    u_hyb_fft,
    err_fft, err_sem, err_hyb,
    t_fft, t_sem, t_hyb
)
