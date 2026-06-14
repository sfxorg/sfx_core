import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(L, X_fft, u_exact, u_fft, X_sem, u_sem, u_exact_sem, 
                       u_hyb_fft, u_ai, err_fft, err_sem, err_hyb, err_ai, 
                       t_fft, t_sem, t_hyb):
    
    plt.switch_backend('Agg') 
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = plt.GridSpec(2, 4, figure=fig)
    cmap = 'viridis'

    # A, B, C, D: Physics Results
    results = [u_exact, u_fft, u_sem, u_hyb_fft]
    titles = ['A. Exact', 'B. FFT', 'C. Pure SEM', 'D. Hybrid Core']
    for i in range(4):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(results[i], extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titles[i], fontweight='bold')

    # E. AI Surrogate
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(u_ai, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax4.set_title('E. AI Surrogate', fontweight='bold')

    # F. Runtime
    ax5 = fig.add_subplot(grid[1, 1])
    ax5.bar(['FFT', 'SEM', 'Hybrid'], [t_fft, t_sem, t_hyb], color=['blue', 'green', 'crimson'])
    ax5.set_title('F. Runtime (s)', fontweight='bold')

    # G. Error Slice
    ax6 = fig.add_subplot(grid[1, 2:])
    x_vec = X_fft[0, :]
    mid = u_fft.shape[0] // 2
    ax6.semilogy(jnp.linspace(0, L, u_sem.shape[0]), err_sem[u_sem.shape[0]//2, :], 'g-', alpha=0.2, lw=5, label='Pure SEM')
    ax6.semilogy(x_vec, err_fft[mid, :], 'b--', lw=1.5, label='Pure FFT')
    ax6.semilogy(x_vec, err_hyb[mid, :], 'r:', lw=2.5, label='Hybrid Core')
    ax6.semilogy(x_vec, err_ai[mid, :], 'k-.', lw=1.5, label='Pure AI Surrogate')
    
    ax6.set_yscale('log')
    ax6.set_ylim(1e-15, 1e1)
    ax6.set_title('G. Accuracy vs AI Surrogate', fontweight='bold')
    ax6.legend()
    
    plt.suptitle('SFX Core Framework: Global Physics vs AI Surrogate', fontsize=15, fontweight='bold')
    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all')
    print("Dashboard saved to sfx_dashboard.png")
