import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_sfx_dashboard(L, X_fft, u_exact, u_fft, X_sem, u_sem, u_exact_sem, 
                       u_hyb_fft, u_ai, u_fv, err_fft, err_sem, err_hyb, 
                       err_ai, err_fv, t_fft, t_sem, t_hyb, t_fv):
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(20, 9), constrained_layout=True)
    grid = plt.GridSpec(2, 4, figure=fig)
    cmap = 'viridis'

    # Physics Results
    data_list = [u_exact, u_fft, u_sem, u_hyb_fft, u_ai, u_fv]
    titles = ['Exact', 'FFT', 'SEM', 'Hybrid', 'AI Surrogate', 'FV Proxy']
    for i, (d, t) in enumerate(zip(data_list, titles)):
        ax = fig.add_subplot(grid[0 if i < 4 else 1, i if i < 4 else i-4])
        ax.imshow(d, extent=[0, L, 0, L], origin='lower', cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t, fontweight='bold')

    # Runtime
    ax_t = fig.add_subplot(grid[1, 2])
    ax_t.bar(['FFT', 'SEM', 'Hyb', 'FV'], [t_fft, t_sem, t_hyb, t_fv], color=['blue', 'green', 'red', 'purple'])
    ax_t.set_title('Runtime (s)', fontweight='bold')

    # Error Comparison
    ax_e = fig.add_subplot(grid[1, 3])
    x_vec = X_fft[0, :]
    mid = u_fft.shape[0] // 2
    
    ax_e.semilogy(jnp.linspace(0, L, u_sem.shape[0]), err_sem[u_sem.shape[0]//2, :], 'g-', alpha=0.2, lw=5, label='Pure SEM')
    ax_e.semilogy(x_vec, err_fft[mid, :], 'b--', lw=1.5, label='Pure FFT')
    ax_e.semilogy(x_vec, err_hyb[mid, :], 'r:', lw=2.5, label='Hybrid Core')
    ax_e.semilogy(x_vec, err_ai[mid, :], 'k-.', lw=1.0, label='AI Surrogate')
    ax_e.semilogy(x_vec, err_fv[mid, :], 'm-', lw=1.0, label='FV Proxy')
    
    ax_e.set_yscale('log')
    ax_e.set_ylim(1e-15, 1e1)
    ax_e.set_title('Accuracy Comparison', fontweight='bold')
    ax_e.legend(loc='upper right', fontsize=7)
    
    plt.savefig("sfx_dashboard.png", dpi=150)
    plt.close('all')
    print("Dashboard saved to sfx_dashboard.png")
