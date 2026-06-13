import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

# Force true double precision and pin execution to your CPU platform
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# =====================================================================
# 1. 2D MATHEMATICAL BASIS CORE GENERATORS
# =====================================================================
def get_sem_diff_matrix_2d(P):
    """Computes exact analytical differentiation matrix for elements."""
    nodes = -jnp.cos(jnp.pi * jnp.arange(P + 1) / P)
    c = jnp.ones(P + 1).at[:].set(2.0).at[-1].set(2.0)
    D = jnp.zeros((P + 1, P + 1))
    for i in range(P + 1):
        for j in range(P + 1):
            if i != j:
                D = D.at[i, j].set((c[i] / c[j]) * ((-1.0)**(i + j)) / (nodes[i] - nodes[j] + 1e-16))
    for i in range(P + 1):
        D = D.at[i, i].set(-jnp.sum(D[i, :]))
    return nodes, D

def generate_tiled_sem_grid_2d(E, P, L):
    """Generates a multi-element continuous grid to eliminate dispersion smearing."""
    nodes_ref, _ = get_sem_diff_matrix_2d(P)
    dx_elem = L / E
    x_coords = []
    for e in range(E):
        x_coords.append((e * dx_elem) + 0.5 * dx_elem * (nodes_ref + 1.0))
    X_elem_1d = jnp.concatenate(x_coords)
    return jnp.meshgrid(X_elem_1d, X_elem_1d)

# =====================================================================
# 2. RUNTIME DYNAMICAL ENGINES (4-SIDED OVERLAPPING SCHWARZ EXCHANGE)
# =====================================================================
@jax.jit
def run_pure_fft_2d(u, ik_x, ik_y, cx, cy, dt):
    u_hat = jnp.fft.fft2(u)
    return u - dt * (cx * jnp.fft.ifft2(u_hat * ik_x).real + cy * jnp.fft.ifft2(u_hat * ik_y).real)

@jax.jit
def run_pure_sem_2d(u, D_matrix, jacobian, cx, cy, dt):
    du_dx = jnp.dot(D_matrix, u) * jacobian
    du_dy = jnp.dot(u, D_matrix.T) * jacobian
    return u - dt * (cx * du_dx + cy * du_dy)

@jax.jit
def run_hybrid_sfx_2d_stable(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right, ik_x, ik_y, D_matrix, jac_sem, cx, cy, dt):
    # 1. Advance Interior Core globally using 2D FFT over full periodic matrix
    u_fft_hat = jnp.fft.fft2(u_fft)
    du_fft_dx = jnp.fft.ifft2(u_fft_hat * ik_x).real
    du_fft_dy = jnp.fft.ifft2(u_fft_hat * ik_y).real
    u_fft_new = u_fft - dt * (cx * du_fft_dx + cy * du_fft_dy)
    
    # 2. Compute Horizontal Ribbon Derivatives (Top/Bottom)
    du_rib_top_dx = jnp.dot(D_matrix, u_rib_top) * jac_sem
    du_rib_top_dy = jnp.fft.ifft(jnp.fft.fft(u_rib_top, axis=1) * ik_y, axis=1).real
    u_rib_top_new = u_rib_top - dt * (cx * du_rib_top_dx + cy * du_rib_top_dy)
    
    du_rib_bot_dx = jnp.dot(D_matrix, u_rib_bot) * jac_sem
    du_rib_bot_dy = jnp.fft.ifft(jnp.fft.fft(u_rib_bot, axis=1) * ik_y, axis=1).real
    u_rib_bot_new = u_rib_bot - dt * (cx * du_rib_bot_dx + cy * du_rib_bot_dy)
    
    # 3. Compute Vertical Ribbon Derivatives (Left/Right)
    du_rib_left_dx = jnp.fft.ifft(jnp.fft.fft(u_rib_left, axis=0) * ik_x, axis=0).real
    du_rib_left_dy = jnp.dot(u_rib_left, D_matrix.T) * jac_sem
    u_rib_left_new = u_rib_left - dt * (cx * du_rib_left_dx + cy * du_rib_left_dy)
    
    du_rib_right_dx = jnp.fft.ifft(jnp.fft.fft(u_rib_right, axis=0) * ik_x, axis=0).real
    du_rib_right_dy = jnp.dot(u_rib_right, D_matrix.T) * jac_sem
    u_rib_right_new = u_rib_right - dt * (cx * du_rib_right_dx + cy * du_rib_right_dy)
    
    # 4. MULTI-ROW 2D SCHWARZ DOMAIN OVERLAP EXCHANGE
    u_rib_top_new = u_rib_top_new.at[0:2, :].set(u_fft_new[-2:, :])
    u_rib_bot_new = u_rib_bot_new.at[-2:, :].set(u_fft_new[0:2, :])
    u_rib_left_new = u_rib_left_new.at[:, -2:].set(u_fft_new[:, 0:2])
    u_rib_right_new = u_rib_right_new.at[:, 0:2].set(u_fft_new[:, -2:])
    
    return u_fft_new, u_rib_top_new, u_rib_bot_new, u_rib_left_new, u_rib_right_new

# =====================================================================
# 3. SPATIAL RESOLUTION CONFIGURATIONS & GRID COUNTING
# =====================================================================
L = 10.0
cx, cy = 1.0, 1.0
dt = 0.00005  
total_time = 12.5  
steps = int(total_time / dt)

# Case 1 Setup
N_fft = 64
dx_fft = L / N_fft
X_fft, Y_fft = jnp.meshgrid(jnp.linspace(0.0, L, N_fft, endpoint=False), jnp.linspace(0.0, L, N_fft, endpoint=False))
ik_x = 1j * 2.0 * jnp.pi * jnp.fft.fftfreq(N_fft, d=dx_fft)[:, None]
ik_y = 1j * 2.0 * jnp.pi * jnp.fft.fftfreq(N_fft, d=dx_fft)[None, :]
total_grid_fft = X_fft.size

# Case 2 Setup (Fair Comparison Mesh: 32 elements * order 4 = 128 matched nodes per axis)
E_sem, P_sem = 32, 4  
X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)
_, D_ref_sem = get_sem_diff_matrix_2d(P_sem)
D_sem_global = jnp.kron(jnp.eye(E_sem), D_ref_sem)
jac_sem_pure = 2.0 / (L / E_sem)
total_grid_sem = X_sem.size

# Case 3 Setup
P_ribbon = 4  
_, D_ref_ribbon = get_sem_diff_matrix_2d(P_ribbon)
jac_sem_hybrid = 2.0 / (L / 16.0)

# Matched Initial Analytical States
u_init_fft = jnp.exp(-((X_fft - 5.0)**2 + (Y_fft - 5.0)**2) / 1.5)
u_init_sem = jnp.exp(-((X_sem - 5.0)**2 + (Y_sem - 5.0)**2) / 1.5)

u_pure_fft = jnp.copy(u_init_fft)
u_pure_sem = jnp.copy(u_init_sem)

# Conformal Overlapping Slices Allocation for the Full 4-Sided Perimeter Frame
u_hyb_fft = jnp.copy(u_init_fft)
u_hyb_rib_bot = jnp.copy(u_init_fft[:P_ribbon+1, :])   
u_hyb_rib_top = jnp.copy(u_init_fft[-P_ribbon-1:, :])  
u_hyb_rib_left = jnp.copy(u_init_fft[:, :P_ribbon+1])   
u_hyb_rib_right = jnp.copy(u_init_fft[:, -P_ribbon-1:]) 
total_grid_hybrid = u_hyb_fft.size + u_hyb_rib_bot.size + u_hyb_rib_top.size + u_hyb_rib_left.size + u_hyb_rib_right.size

# =====================================================================
# 4. SIMULATION BENCHMARK TIME LOOP
# =====================================================================
print(f"Running Upgraded Matched 2D Framework for {steps} steps until t=12.5s...")
t0 = time.time()
for _ in range(steps):
    u_pure_fft = run_pure_fft_2d(u_pure_fft, ik_x, ik_y, cx, cy, dt)
u_pure_fft.block_until_ready()
t_fft = time.time() - t0

t0 = time.time()
for _ in range(steps):
    u_pure_sem = run_pure_sem_2d(u_pure_sem, D_sem_global, jac_sem_pure, cx, cy, dt)
u_pure_sem.block_until_ready()
t_sem = time.time() - t0

t0 = time.time()
for _ in range(steps):
    u_hyb_fft, u_hyb_rib_top, u_hyb_rib_bot, u_hyb_rib_left, u_hyb_rib_right = run_hybrid_sfx_2d_stable(
        u_hyb_fft, u_hyb_rib_top, u_hyb_rib_bot, u_hyb_rib_left, u_hyb_rib_right, 
        ik_x, ik_y, D_ref_ribbon, jac_sem_hybrid, cx, cy, dt
    )
u_hyb_fft.block_until_ready()
t_hybrid = time.time() - t0

# Compute exact shifted analytical target profiles
X_shifted = (X_fft - cx * total_time) % L
Y_shifted = (Y_fft - cy * total_time) % L
u_exact_shifted = jnp.exp(-((X_shifted - 5.0)**2 + (Y_shifted - 5.0)**2) / 1.5)

X_sem_shifted = (X_sem - cx * total_time) % L
Y_sem_shifted = (Y_sem - cy * total_time) % L
u_exact_sem_shifted = jnp.exp(-((X_sem_shifted - 5.0)**2 + (Y_sem_shifted - 5.0)**2) / 1.5)

err_fft = jnp.abs(u_pure_fft - u_exact_shifted)
err_sem = jnp.abs(u_pure_sem - u_exact_sem_shifted)
err_hyb = jnp.abs(u_hyb_fft - u_exact_shifted)

u_hybrid_full = jnp.copy(u_hyb_fft)

# Print Text Metrics ledger to terminal
print("\n" + "="*55)
print(f"       2D REBALANCED CORE ACCURACY REPORT (t={total_time}s) ")
print("="*55)
print(f"Case 1: Pure 2D FFT Max Error        : {jnp.max(err_fft):.2e}")
print(f"Case 2: Pure 2D SEM Max Error        : {jnp.max(err_sem):.2e}")
print(f"Case 3: Inverted Hybrid Max Error    : {jnp.max(err_hyb):.2e}")
print("-"*55)
print(f"Case 1 Baseline Run Time             : {t_fft:.4f} seconds")
print(f"Case 2 Tiled Mesh Run Time           : {t_sem:.4f} seconds")
print(f"Case 3 Hybrid Engine Run Time        : {t_hybrid:.4f} seconds")
print("-"*55)
print(f"Case 1 Total Allocated Grid Nodes    : {total_grid_fft:,} DoF")
print(f"Case 2 Total Allocated Grid Nodes    : {total_grid_sem:,} DoF")
print(f"Case 3 Total Allocated Grid Nodes    : {total_grid_hybrid:,} DoF (Core + 4 Schwarz Overlaps)")
print("="*55)

# =====================================================================
# 5. GENERATE DIAGNOSTIC COMPOSITE DASHBOARD
# =====================================================================
fig = plt.figure(figsize=(15, 9))
grid = plt.GridSpec(2, 3, figure=fig)
cmap_choice = 'viridis'

ax0 = fig.add_subplot(grid[0, 0])
ax0.imshow(u_exact_shifted, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
ax0.set_title('A. Exact Shifted Target (t=12.5s)', fontsize=10, fontweight='bold')

ax1 = fig.add_subplot(grid[0, 1])
ax1.imshow(u_pure_fft, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
ax1.set_title('B. Case 1: Pure Global 2D FFT', fontsize=10, fontweight='bold')

ax2 = fig.add_subplot(grid[0, 2])
ax2.imshow(u_pure_sem, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
ax2.set_title('C. Case 2: Pure Tiled SEM (Moving)', fontsize=10, fontweight='bold')

ax3 = fig.add_subplot(grid[1, 0])
ax3.imshow(u_hybrid_full, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
ax3.set_title('D. Case 3: Inverted Hybrid (SFX Schwarz Overlap)', fontsize=10, fontweight='bold')

ax4 = fig.add_subplot(grid[1, 1])
bars = ax4.bar(['FFT', 'SEM', 'Hybrid'], [t_fft, t_sem, t_hybrid], color=['blue', 'green', 'crimson'], width=0.4, edgecolor='k')
ax4.set_title('E. Runtime Speed Benchmark (s)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Seconds')

# FIXED: Increased upper boundary ceiling to 180s to gracefully insulate bar data text labels
ax4.set_ylim(0, 180)

for bar in bars:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05, f"{bar.get_height():.2f}s", ha='center', va='bottom', fontsize=9, fontweight='bold')

ax5 = fig.add_subplot(grid[1, 2])
x_vector = X_fft[0, :]
mid_idx_fft = N_fft // 2

ax5.plot(x_vector, err_fft[mid_idx_fft, :], 'b-', linewidth=2, label='Pure FFT')
ax5.plot(jnp.linspace(0, L, u_pure_sem.shape[0]), err_sem[u_pure_sem.shape[0]//2, :], 'g--', alpha=0.6, label='Pure SEM')
ax5.plot(x_vector, err_hyb[mid_idx_fft, :], 'r:', linewidth=2.5, label='Hybrid Core')
ax5.set_yscale('log')
ax5.set_ylim(1e-12, 1e2)
ax5.set_title('F. Center Slice Absolute Error Log', fontsize=10, fontweight='bold')
ax5.set_xlabel('Spatial Axis (x)')
ax5.set_ylabel('Absolute Error Magnitude')
ax5.legend(loc='lower left', fontsize=8)

for ax in [ax0, ax1, ax2, ax3]:
    ax.set_xlabel('Spatial Axis (x)', fontsize=8)
    ax.set_ylabel('Spatial Axis (y)', fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.3)

plt.suptitle(f'SFX Core Framework: 4-Sided Overlapping Schwarz Report', fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig("sfx_2d_master_benchmark_report.png", dpi=150)
print("Complete master 4-sided overlapping Schwarz report successfully compiled and saved.")
plt.show()
