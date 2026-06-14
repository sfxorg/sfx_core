import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(L, X_fft, u_exact, u_fft, X_sem, u_sem, u_exact_sem, 
                       u_hyb_fft, u_ai, u_fv, err_fft, err_sem, err_hyb, 
                       err_ai, err_fv, t_fft, t_sem, t_hyb, t_fv):
    
    # 1. Force Agg backend to prevent GUI 'grab' crashes
    plt.switch_backend('Agg') 
    
    # 2. 2x4 Layout: Top row for 4 Physics results, Bottom row for AI/FV/Runtime/Errors
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = plt.GridSpec(2, 4, figure=fig)
    cmap = 'viridis'

    # A, B, C, D: Physics Results (Top Row)
    data_list = [u_exact, u_fft, u_sem, u_hyb_fft]
    titles = ['A. Exact', 'B. Pure FFT', 'C. Pure SEM', 'D. Hybrid Core']
    for i in range(4):
        ax = fig.add_subplot(grid[0, i])
        ax.imshow(data_list[i], extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(titles[i], fontweight='bold')

    # E. AI Surrogate Result
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(u_ai, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax4.set_title('E. AI Surrogate', fontweight='bold')

    # F. FV Proxy Result
    ax5 = fig.add_subplot(grid[1, 1])
    ax5.imshow(u_fv, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax5.set_title('F. FV Proxy', fontweight='bold')

    # Runtime Benchmarks
    ax6 = fig.add_subplot(grid[1, 2])
    ax6.bar(['FFT', 'SEM', 'Hyb', 'FV'], [t_fft, t_sem, t_hyb, t_fv], color=['blue', 'green', 'red', 'purple'])
    ax6.set_title('G. Runtime (s)', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)

    # H. Error Comparison (The Headliner)
    ax7 = fig.add_subplot(grid[1, 3])
    x_vec = X_fft[0, :]
    mid = u_fft.shape[0] // 2
    
    # Plotting order: SEM(Base) -> FFT -> Hybrid -> AI -> FV
    ax7.semilogy(jnp.linspace(0, L, u_sem.shape[0]), err_sem[u_sem.shape[0]//2, :], 
                 'g-', alpha=0.2, linewidth=5, label='Pure SEM')
    ax7.semilogy(x_vec, err_fft[mid, :], 'b--', lw=1.2, label='Pure FFT')
    ax7.semilogy(x_vec, err_hyb[mid, :], 'r:', lw=2.0, label='Hybrid Core')
    ax7.semilogy(x_vec, err_ai[mid, :], 'k-.', lw=1.0, label='AI Surrogate')
    ax7.semilogy(x_vec, err_fv[mid, :], 'm-', lw=1.0, label='FV Proxy')
    
    ax7.set_yscale('log')
    ax7.set_ylim(1e-15, 1e1)
    ax7.set_title('H. Accuracy Comparison (Log)', fontweight='bold')
    ax7.legend(loc='upper right', fontsize=7)
    
    plt.suptitle('SFX DCORE Framework: Physics vs. Surrogate Comparison', fontsize=16, fontweight='bold')
    
    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all')
    print("Dashboard saved to sfx_dashboard.png")
