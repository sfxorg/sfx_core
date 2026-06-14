import jax
import jax.numpy as jnp

@jax.jit
def run_hybrid_sfx_2d_stable(
    u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right, 
    ik_x, ik_y, D_matrix, jac_sem, cx, cy, dt
):
    # Penalty constant: higher values enforce stricter coupling.
    # If NaN persists, reduce this to 1.0/dt or lower.
    penalty = 1.0 / dt

    def rhs(u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right):
        
        # FFT interior derivatives
        u_fft_hat = jnp.fft.fft2(u_fft)
        d_fft_dt = -(cx * jnp.fft.ifft2(u_fft_hat * ik_x).real + 
                     cy * jnp.fft.ifft2(u_fft_hat * ik_y).real)

        # Horizontal ribbons (with Penalty Force)
        def update_horizontal_spatial(rib, fft_boundary, side):
            du_dx = jnp.dot(D_matrix, rib) * jac_sem
            du_dy = jnp.fft.ifft(jnp.fft.fft(rib, axis=1) * ik_y, axis=1).real
            d_dt = -(cx * du_dx + cy * du_dy)
            
            # Penalty coupling: Add force to ribbon boundary instead of forcing set()
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
    
    def add_k(state, k, factor):
        return tuple(s + factor * dk for s, dk in zip(state, k))

    k2 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k1, 0.5 * dt))
    k3 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k2, 0.5 * dt))
    k4 = rhs(*add_k((u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right), k3, dt))

    # Combine stages
    new_state = []
    for i in range(5):
        new_state.append(
            (u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right)[i] + 
            (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
        )
        
    return tuple(new_state)
