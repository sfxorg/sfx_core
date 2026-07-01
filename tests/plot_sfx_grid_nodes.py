import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax.numpy as jnp
import matplotlib.pyplot as plt
from geometry.sem_grids import (generate_ribbon_grid)

L = 10.0
N_fft = 64
P_ribbon = 4

# -------------------------------
# FFT GRID
# -------------------------------
X_fft, Y_fft = jnp.meshgrid(
    jnp.linspace(0, L, N_fft, endpoint=False),
    jnp.linspace(0, L, N_fft, endpoint=False)
)
total_nodes_fft = X_fft.size

# -------------------------------
# HYBRID RIBBON GRIDS
# -------------------------------
X_hyb_1d = generate_ribbon_grid(P_ribbon, L, 16)

# Horizontal ribbons
y_bot = 0.5 * (L/16.0) * (jnp.linspace(-1, 1, P_ribbon+1) + 1.0)
y_top = L - y_bot
X_rib_bot, Y_rib_bot = jnp.meshgrid(X_hyb_1d, y_bot)
X_rib_top, Y_rib_top = jnp.meshgrid(X_hyb_1d, y_top)

# Vertical ribbons
x_left = 0.5 * (L/16.0) * (jnp.linspace(-1, 1, P_ribbon+1) + 1.0)
x_right = L - x_left
X_rib_left, Y_rib_left = jnp.meshgrid(x_left, X_hyb_1d)
X_rib_right, Y_rib_right = jnp.meshgrid(x_right, X_hyb_1d)

# Interior FFT core (trimmed)
X_core_flat = X_fft[:, P_ribbon+1:-P_ribbon-1].ravel()
Y_core_flat = Y_fft[:, P_ribbon+1:-P_ribbon-1].ravel()

total_nodes_hybrid = (
    X_core_flat.size + X_rib_bot.size + X_rib_top.size + 
    X_rib_left.size + X_rib_right.size
)

# -------------------------------
# PLOTTING
# -------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Panel A: FFT
axs[0].plot(X_fft.ravel(), Y_fft.ravel(), 'b.', markersize=0.8, alpha=0.4)
axs[0].set_title(f'A. Pure FFT Grid\n({total_nodes_fft:,} DoF)', fontsize=11, fontweight='bold')

# Panel B: Hybrid
axs[1].plot(X_core_flat, Y_core_flat, 'r.', markersize=0.5, alpha=0.2)
axs[1].plot(X_rib_bot.ravel(), Y_rib_bot.ravel(), 'k.', markersize=1.2, alpha=0.7)
axs[1].plot(X_rib_top.ravel(), Y_rib_top.ravel(), 'k.', markersize=1.2, alpha=0.7)
axs[1].plot(X_rib_left.ravel(), Y_rib_left.ravel(), 'k.', markersize=1.2, alpha=0.7)
axs[1].plot(X_rib_right.ravel(), Y_rib_right.ravel(), 'k.', markersize=1.2, alpha=0.7)
axs[1].set_title(f'B. Hybrid Grid\n({total_nodes_hybrid:,} DoF)', fontsize=11, fontweight='bold')

for ax in axs:
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, linestyle=':', alpha=0.2)

plt.suptitle('SFX Core Framework: Grid Structures', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("sfx_2d_grid_structures.png", dpi=150)
plt.show()
