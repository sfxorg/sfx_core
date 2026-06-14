import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(
    L, X_fft, u_exact_shifted, u_fft, 
    X_sem, u_sem, u_exact_sem_shifted, 
    u_hyb_fft, err_fft, err_sem, err_hyb, 
    t_fft, t_sem, t_hyb 
):
    # 1. Force Agg backend to prevent Tkinter 'grab' crashes
    plt.switch_backend('Agg')
    
    # 2. Use constrained_layout=True instead of tight_layout()
    # This automatically manages spacing without deleting overlapping axes
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = plt.GridSpec(2, 3, figure=fig)
    cmap_choice = 'viridis'

    # A. Exact
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.imshow(u_exact_shifted, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
    ax0.set_title('A. Exact Target (t=12.5s)')

    # B. FFT
    ax1 = fig.add_subplot(grid[0, 1])
    ax1.imshow(u_fft, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
    ax1.set_title('B. Case 1: Pure Global 2D FFT')

    # C. SEM
    ax2 = fig.add_subplot(grid[0, 2])
    ax2.imshow(u_sem, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
    ax2.set_title('C. Case 2: Pure Tiled SEM')

    # D. Hybrid
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.imshow(u_hyb_fft, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
    ax3.set_title('D. Case 3: Inverted Hybrid')

    # E. Runtime bars
    ax4 = fig.add_subplot(grid[1, 1])
    bars = ax4.bar(['FFT', 'SEM', 'Hybrid'], [t_fft, t_sem, t_hyb], color=['blue', 'green', 'crimson'], width=0.4, edgecolor='k')
    ax4.set_title('E. Runtime Speed (s)')
    ax4.set_ylabel('Seconds')
    for bar in bars:
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05, f"{bar.get_height():.2f}s", ha='center', va='bottom', fontsize=9)

    # F. Error slice
    ax5 = fig.add_subplot(grid[1, 2])
    x_vector = X_fft[0, :]
    mid_idx = u_fft.shape[0] // 2
    
    # Plot SEM first (transparency), then precision lines on top
    ax5.semilogy(jnp.linspace(0, L, u_sem.shape[0]), err_sem[u_sem.shape[0]//2, :], 
                 'g-', alpha=0.3, linewidth=4, label='Pure SEM')
    ax5.semilogy(x_vector, err_fft[mid_idx, :], 'b--', linewidth=1.5, label='Pure FFT')
    ax5.semilogy(x_vector, err_hyb[mid_idx, :], 'r:', linewidth=2.0, label='Hybrid Core')
    
    ax5.set_ylim(1e-15, 1e1)
    ax5.set_title('F. Center Slice Error (Log)')
    ax5.legend(loc='upper right', fontsize=8)

    # Clean up formatting
    for ax in [ax0, ax1, ax2, ax3]:
        ax.grid(True, linestyle=':', alpha=0.3)
        
    plt.suptitle('SFX Core Framework: 4-Sided Overlapping Schwarz Report', fontsize=13, fontweight='bold')
    
    # Save instead of show
    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all')
    print("Dashboard saved to sfx_dashboard.png")
