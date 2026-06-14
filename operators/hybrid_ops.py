import jax
import jax.numpy as jnp

@jax.jit
def run_hybrid_sfx_2d_stable(
    u_fft, u_rib_top, u_rib_bot, u_rib_left, u_rib_right,
    ik_x, ik_y, D_matrix, jac_sem, cx, cy, dt
):
    # FFT interior
    u_fft_hat = jnp.fft.fft2(u_fft)
    du_fft_dx = jnp.fft.ifft2(u_fft_hat * ik_x).real
    du_fft_dy = jnp.fft.ifft2(u_fft_hat * ik_y).real
    u_fft_new = u_fft - dt * (cx * du_fft_dx + cy * du_fft_dy)

    # Horizontal ribbons
    def update_horizontal(rib):
        du_dx = jnp.dot(D_matrix, rib) * jac_sem
        du_dy = jnp.fft.ifft(jnp.fft.fft(rib, axis=1) * ik_y, axis=1).real
        return rib - dt * (cx * du_dx + cy * du_dy)

    u_rib_top_new = update_horizontal(u_rib_top)
    u_rib_bot_new = update_horizontal(u_rib_bot)

    # Vertical ribbons
    def update_vertical(rib):
        du_dx = jnp.fft.ifft(jnp.fft.fft(rib, axis=0) * ik_x, axis=0).real
        du_dy = jnp.dot(rib, D_matrix.T) * jac_sem
        return rib - dt * (cx * du_dx + cy * du_dy)

    u_rib_left_new = update_vertical(u_rib_left)
    u_rib_right_new = update_vertical(u_rib_right)

    # Schwarz overlap exchange
    u_rib_top_new = u_rib_top_new.at[0:2, :].set(u_fft_new[-2:, :])
    u_rib_bot_new = u_rib_bot_new.at[-2:, :].set(u_fft_new[0:2, :])
    u_rib_left_new = u_rib_left_new.at[:, -2:].set(u_fft_new[:, 0:2])
    u_rib_right_new = u_rib_right_new.at[:, 0:2].set(u_fft_new[:, -2:])

    return (
        u_fft_new,
        u_rib_top_new,
        u_rib_bot_new,
        u_rib_left_new,
        u_rib_right_new
    )
