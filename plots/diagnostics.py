import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(L, X_fft, u_exact_shifted, u_fft, X_sem, u_sem, 
                       u_exact_sem_shifted, u_hyb_fft, err_fft, err_sem, 
                       err_hyb, t_fft, t_sem, t_hyb):
    
    # Force Agg backend to prevent GUI 'grab' crashes
    plt.switch_backend('Agg') 
    
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = plt.GridSpec(2, 3, figure=fig)
    cmap_choice = 'viridis'

    # Images A, B, C, D
    # (Keep your existing imshow logic here)
    for i, (data, title) in enumerate(zip(
        [u_exact_shifted, u_fft, u_sem, u_hyb_fft], 
        ['A. Exact', 'B. FFT', 'C. Pure SEM', 'D. Hybrid']
    )):
        ax = fig.add_subplot(grid[0 if i < 3 else 1, i % 3 if i < 3 else 0])
        ax.imshow(data, extent=[0, L, 0, L], origin='lower', cmap=cmap_choice, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='bold')

    # E. Runtime bars
    ax4 = fig.add_subplot(grid[1, 1])
    bars = ax4.bar(['FFT', 'SEM', 'Hybrid'], [t_fft, t_sem, t_hyb], 
                   color=['blue', 'green', 'crimson'], width=0.4, edgecolor='k')
    ax4.set_title('E. Runtime Speed (s)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Seconds')

    # F. Error slice - FIXED Z-ORDER
    ax5 = fig.add_subplot(grid[1, 2])
    x_vec_fft = X_fft[0, :]
    x_vec_sem = jnp.linspace(0, L, u_sem.shape[0])
    
    # 1. Plot SEM first (Low alpha/Thick) - The 'Baseline'
    ax5.semilogy(x_vec_sem, err_sem[u_sem.shape[0]//2, :], 
                 'g-', alpha=0.3, linewidth=4, label='Pure SEM')
    
    # 2. Plot FFT/Hybrid on TOP (Precision lines)
    ax5.semilogy(x_vec_fft, err_fft[u_fft.shape[0]//2, :], 
                 'b--', linewidth=1.5, label='Pure FFT')
    ax5.semilogy(x_vec_fft, err_hyb[u_fft.shape[0]//2, :], 
                 'r:', linewidth=2.0, label='Hybrid Core')
    
    ax5.set_yscale('log')
    ax5.set_ylim(1e-15, 1e1)
    ax5.set_title('F. Center Slice Error (Log)', fontsize=10, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)

    plt.suptitle('SFX Core Framework: 4-Sided Overlapping Schwarz Report', fontsize=13, fontweight='bold')
    
    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all')
    print("Dashboard saved to sfx_dashboard.png")
