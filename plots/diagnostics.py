import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(L, X_fft, u_init, u_exact, u_fft, u_hyb_std, u_hyb_sinc, u_fv, err_fft, err_hyb_std, err_hyb_sinc, err_fv, t_fft, t_hyb_std, t_hyb_sinc, t_fv):
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = plt.GridSpec(2, 4, figure=fig)
    cmap = 'viridis'

    # A. Exact (Final State) with Initial State Overlay
    ax0 = fig.add_subplot(grid[0, 0])
    ax0.imshow(u_exact, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax0.contour(u_init, extent=[0, L, 0, L], origin='lower', colors='white', linestyles='dashed', levels=[0.5], alpha=0.8)
    ax0.set_title('A. Exact (Final + Init Cont.)', fontweight='bold')

    # B, C: Physics Results (FFT and Hybrid Std)
    results = [u_fft, u_hyb_std]
    titles = ['B. FFT', 'C. Hybrid Std']
    for i in range(2):
        ax = fig.add_subplot(grid[0, i+1])
        ax.imshow(results[i], extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titles[i], fontweight='bold')

    # E. Hybrid Sinc Result
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(u_hyb_sinc, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax4.set_title('D. Hybrid Sinc', fontweight='bold')

    # F. FV Proxy Result
    ax5 = fig.add_subplot(grid[1, 1])
    ax5.imshow(u_fv, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax5.set_title('E. FV Proxy', fontweight='bold')

    # G. Runtime
    ax6 = fig.add_subplot(grid[1, 2])
    ax6.bar(['FFT', 'Hyb-Std', 'Hyb-Sinc', 'FV'], [t_fft, t_hyb_std, t_hyb_sinc, t_fv], color=['blue', 'red', 'orange', 'purple'])
    ax6.set_title('F. Runtime (s)', fontweight='bold')

    # H. Error Comparison
    ax7 = fig.add_subplot(grid[1, 3])
    x_vec = X_fft[0, :]
    mid = u_fft.shape[0] // 2
    
    ax7.semilogy(x_vec, err_fft[mid, :], 'b--', lw=1.5, label='Pure FFT')
    ax7.semilogy(x_vec, err_hyb_std[mid, :], 'r:', lw=2.5, label='Hybrid Std')
    ax7.semilogy(x_vec, err_hyb_sinc[mid, :], 'y-.', lw=1.5, label='Hybrid Sinc')
    ax7.semilogy(x_vec, err_fv[mid, :], 'm-', lw=1.0, label='FV Proxy')
    
    ax7.set_yscale('log')
    ax7.set_ylim(1e-15, 1e1)
    ax7.set_title('G. Accuracy Comparison', fontweight='bold')
    ax7.legend(loc='lower right', fontsize=7)

    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all')
    print("Dashboard saved to sfx_dashboard.png")
