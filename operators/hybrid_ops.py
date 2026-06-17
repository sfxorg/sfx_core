import jax
import jax.numpy as jnp

# --- 1. STANDARD PENALTY HYBRID ---
@jax.jit
def run_hybrid_sfx_2d_standard(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right, ik_x, ik_y, D_matrix, jac_sem, cx, cy, dt):
    """Original Penalty-based hybrid solver."""
    penalty = 1.0 / dt

    def rhs(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right):
        # FFT interior derivatives
        u_fft_hat = jnp.fft.fft2(u_fft)
        d_fft_dt = -(cx * jnp.fft.ifft2(u_fft_hat * ik_x).real + cy * jnp.fft.ifft2(u_fft_hat * ik_y).real)

        # Horizontal ribbons (with Penalty Force)
        def update_horizontal_spatial(rib, fft_boundary, side):
            du_dx = jnp.dot(D_matrix, rib) * jac_sem
            du_dy = jnp.fft.ifft(jnp.fft.fft(rib, axis=1) * ik_y, axis=1).real
            d_dt = -(cx * du_dx + cy * du_dy)
            if side == 'top':
                d_dt = d_dt.at[0:2, :].add(penalty * (fft_boundary[-2:, :] - rib[0:2, :]))
            else: # bottom
                d_dt = d_dt.at[-2:, :].add(penalty * (fft_boundary[0:2, :] - rib[-2:, :]))
            return d_dt

        d_rib_top_dt = update_horizontal_spatial(u_rib_top, u_fft, 'top')
        d_rib_bot_dt = update_horizontal_spatial(u_rib_bot, u_fft, 'bot')

        # Vertical ribbons (with Penalty Force)
        def update_vertical_spatial(rib, fft_boundary, side):
            du_dx = jnp.fft.ifft(jnp.fft.fft(rib, axis=0) * ik_x, axis=0).real
            du_dy = jnp.dot(rib, D_matrix.T) * jac_sem
            d_dt = -(cx * du_dx + cy * du_dy)
            if side == 'left':
                d_dt = d_dt.at[:, -2:].add(penalty * (fft_boundary[:, 0:2] - rib[:, -2:]))
            else: # right
                d_dt = d_dt.at[:, 0:2].add(penalty * (fft_boundary[:, -2:] - rib[:, 0:2]))
            return d_dt

        d_rib_left_dt = update_vertical_spatial(u_rib_left, u_fft, 'left')
        d_rib_right_dt = update_vertical_spatial(u_rib_right, u_fft, 'right')
        return d_fft_dt, d_rib_top_dt, d_rib_bot_dt, d_rib_left_dt, d_rib_right_dt

    # RK4 Stages
    k1 = rhs(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right)
    def add_k(state, k, factor): return tuple(s + factor * dk for s, dk in zip(state, k))
    k2 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k1, 0.5 * dt))
    k3 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k2, 0.5 * dt))
    k4 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k3, dt))
    
    return tuple((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right)[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(5))


# --- 2. SINC-COUPLED HYBRID ---
@jax.jit
def run_hybrid_sfx_2d_sinc(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right, ik_x, ik_y, D_matrix, jac_sem, cx, cy, dt):
    """Hybrid solver using Sinc-kernel projection coupling (Fixed Indexing)."""
    penalty = 1.0 / dt
    
    def sinc_coupling(data, target_shape, bandwidth=1.0):
        # Perform 1D interpolation while preserving the 2D structure
        # data.shape is (N, N), we want output (target_rows, N)
        sinc_kernel = jnp.sinc(bandwidth * (jnp.arange(data.shape[0]) - data.shape[0]/2))
        # Result of sum is (N,), we reshape/tile to (target_rows, N)
        res_1d = jnp.sum(data * sinc_kernel[:, None], axis=0) 
        return jnp.tile(res_1d, (target_shape[0], 1))

    def rhs(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right):
        u_fft_hat = jnp.fft.fft2(u_fft)
        d_fft_dt = -(cx * jnp.fft.ifft2(u_fft_hat * ik_x).real + cy * jnp.fft.ifft2(u_fft_hat * ik_y).real)

        # Horizontal (Ribbon shape is e.g., (2, N))
        def update_horizontal_spatial(rib, fft_boundary, side):
            du_dx = jnp.dot(D_matrix, rib) * jac_sem
            du_dy = jnp.fft.ifft(jnp.fft.fft(rib, axis=1) * ik_y, axis=1).real
            d_dt = -(cx * du_dx + cy * du_dy)
            
            coupled_fft = sinc_coupling(fft_boundary, rib.shape)
            if side == 'top':
                d_dt = d_dt.at[0:2, :].add(penalty * (coupled_fft[0:2, :] - rib[0:2, :]))
            else:
                d_dt = d_dt.at[-2:, :].add(penalty * (coupled_fft[-2:, :] - rib[-2:, :]))
            return d_dt

        d_rib_top_dt = update_horizontal_spatial(u_rib_top, u_fft, 'top')
        d_rib_bot_dt = update_horizontal_spatial(u_rib_bot, u_fft, 'bot')

        # Vertical (Ribbon shape is e.g., (N, 2))
        def update_vertical_spatial(rib, fft_boundary, side):
            du_dx = jnp.fft.ifft(jnp.fft.fft(rib, axis=0) * ik_x, axis=0).real
            du_dy = jnp.dot(rib, D_matrix.T) * jac_sem
            d_dt = -(cx * du_dx + cy * du_dy)
            
            # Transpose domain for vertical sinc application, then transpose back
            coupled_fft = sinc_coupling(fft_boundary.T, rib.T.shape).T
            if side == 'left':
                d_dt = d_dt.at[:, -2:].add(penalty * (coupled_fft[:, -2:] - rib[:, -2:]))
            else:
                d_dt = d_dt.at[:, 0:2].add(penalty * (coupled_fft[:, 0:2] - rib[:, 0:2]))
            return d_dt

        d_rib_left_dt = update_vertical_spatial(u_rib_left, u_fft, 'left')
        d_rib_right_dt = update_vertical_spatial(u_rib_right, u_fft, 'right')
        return d_fft_dt, d_rib_top_dt, d_rib_bot_dt, d_rib_left_dt, d_rib_right_dt

    # RK4 Stages (same as before)
    k1 = rhs(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right)
    def add_k(state, k, factor): return tuple(s + factor * dk for s, dk in zip(state, k))
    k2 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k1, 0.5 * dt))
    k3 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k2, 0.5 * dt))
    k4 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k3, dt))
    
    return tuple((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right)[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(5))
