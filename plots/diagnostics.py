import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(
    L,
    X_fft, u_exact_shifted, u_fft,
    X_sem, u_sem, u_exact_sem_shifted,
    u_hyb_fft,
    err_fft, err_sem, err_hyb,
    t_fft, t_sem, t_hyb
):
    fig = plt.figure(figsize=(15, 9))
    grid = plt.GridSpec(2, 3, figure=fig)
    cmap_choice = 'viridis'

    # A. Exact
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.imshow(u_exact_shifted, extent=[0, L, 0, L], origin='lower',
               cmap=cmap_choice, vmin=0, vmax=1)
    ax0.set_title('A. Exact Shifted Target (t=12.5s)', fontsize=10, fontweight='bold')

    # B. FFT
    ax1 = fig.add_subplot(grid[0, 1])
    ax1.imshow(u_fft, extent=[0, L, 0, L], origin='lower',
               cmap=cmap_choice, vmin=0, vmax=1)
    ax1.set_title('B. Case 1: Pure Global 2D FFT', fontsize=10, fontweight='bold')

    # C. SEM
    ax2 = fig.add_subplot(grid[0, 2])
    ax2.imshow(u_sem, extent=[0, L, 0, L], origin='lower',
               cmap=cmap_choice, vmin=0, vmax=1)
    ax2.set_title('C. Case 2: Pure Tiled SEM (Moving)', fontsize=10, fontweight='bold')

    # D. Hybrid
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.imshow(u_hyb_fft, extent=[0, L, 0, L], origin='lower',
               cmap=cmap_choice, vmin=0, vmax=1)
    ax3.set_title('D. Case 3: Inverted Hybrid (SFX Schwarz Overlap)', fontsize=10, fontweight='bold')

    # E. Runtime bars
    ax4 = fig.add_subplot(grid[1, 1])
    bars = ax4.bar(['FFT', 'SEM', 'Hybrid'], [t_fft, t_sem, t_hyb],
                   color=['blue', 'green', 'crimson'], width=0.4, edgecolor='k')
    ax4.set_title('E. Runtime Speed Benchmark (s)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Seconds')
    ax4.set_ylim(0, max(t_sem, t_hyb) * 1.3)

    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05,
                 f"{bar.get_height():.2f}s", ha='center', va='bottom',
                 fontsize=9, fontweight='bold')

    # F. Error slice
    ax5 = fig.add_subplot(grid[1, 2])
    x_vector = X_fft[0, :]
    mid_fft = u_fft.shape[0] // 2
    mid_sem = u_sem.shape[0] // 2

    ax5.plot(x_vector, err_fft[mid_fft, :], 'b-', linewidth=2, label='Pure FFT')
    ax5.plot(jnp.linspace(0, L, u_sem.shape[0]), err_sem[mid_sem, :],
             'g--', alpha=0.6, label='Pure SEM')
    ax5.plot(x_vector, err_hyb[mid_fft, :], 'r:', linewidth=2.5, label='Hybrid Core')

    ax5.set_yscale('log')
    ax5.set_ylim(1e-12, 1e2)
    ax5.set_title('F. Center Slice Absolute Error Log', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Spatial Axis (x)')
    ax5.set_ylabel('Absolute Error Magnitude')
    ax5.legend(loc='lower left', fontsize=8)

    # Common formatting
    for ax in [ax0, ax1, ax2, ax3]:
        ax.set_xlabel('Spatial Axis (x)', fontsize=8)
        ax.set_ylabel('Spatial Axis (y)', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.3)

    plt.suptitle('SFX Core Framework: 4-Sided Overlapping Schwarz Report',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
