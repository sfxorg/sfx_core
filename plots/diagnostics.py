import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(
    L, X_fft, u_exact_shifted, u_fft, 
    X_sem, u_sem, u_exact_sem_shifted, 
    u_hyb_fft, err_fft, err_sem, err_hyb, 
    t_fft, t_sem, t_hyb 
):
    # Use Agg (non-interactive) to prevent GUI 'grab' crashes
    plt.switch_backend('Agg') 
    
    fig = plt.figure(figsize=(15, 9))
    grid = plt.GridSpec(2, 3, figure=fig)
    cmap_choice = 'viridis'

    # A, B, C, D: Image plots (kept as you had them)
    # ... [Keep your existing ax0, ax1, ax2, ax3 imshow code here] ...

    # E. Runtime bars
    ax4 = fig.add_subplot(grid[1, 1])
    bars = ax4.bar(['FFT', 'SEM', 'Hybrid'], [t_fft, t_sem, t_hyb], color=['blue', 'green', 'crimson'], width=0.4, edgecolor='k')
    ax4.set_title('E. Runtime Speed Benchmark (s)', fontsize=10, fontweight='bold')
    
    # F. Error slice (FIXED ORDER)
    ax5 = fig.add_subplot(grid[1, 2])
    x_vector = X_fft[0, :]
    mid_fft = u_fft.shape[0] // 2
    mid_sem = u_sem.shape[0] // 2
    
    # PLOT SEM FIRST (Thick/Transparent so it doesn't block precision lines)
    ax5.plot(jnp.linspace(0, L, u_sem.shape[0]), err_sem[mid_sem, :], 
             'g-', alpha=0.3, linewidth=4, label='Pure SEM')
    
    # PLOT FFT and HYBRID on top (Precision lines)
    ax5.plot(x_vector, err_fft[mid_fft, :], 'b--', linewidth=1.5, label='Pure FFT')
    ax5.plot(x_vector, err_hyb[mid_fft, :], 'r:', linewidth=2.0, label='Hybrid Core')
    
    ax5.set_yscale('log')
    ax5.set_ylim(1e-15, 1e1)
    ax5.set_title('F. Center Slice Absolute Error (Log)', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Spatial Axis (x)')
    ax5.set_ylabel('Absolute Error Magnitude')
    ax5.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('SFX Core Framework: 4-Sided Overlapping Schwarz Report', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # CRITICAL: Save to disk instead of plt.show()
    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all') # This kills the Tkinter backend properly
    print("Dashboard saved to sfx_dashboard.png")
