import jax
import jax.numpy as jnp

# ============================================================
# STANDARD PENALTY HYBRID
# ============================================================

@jax.jit
def run_hybrid_sfx_2d_standard(
    u_fft,
    u_rib_top,
    u_rib_bot,
    u_rib_left,
    u_rib_right,
    ik_x,
    ik_y,
    D_matrix,
    jac_sem,
    cx,
    cy,
    dt,
    dx
):

    penalty = 1.0 / dt

    def rhs(
        u_fft,
        u_rib_top,
        u_rib_bot,
        u_rib_left,
        u_rib_right,
    ):

        u_fft_hat = jnp.fft.fft2(u_fft)

        d_fft_dt = -(
            cx * jnp.fft.ifft2(u_fft_hat * ik_x).real
            + cy * jnp.fft.ifft2(u_fft_hat * ik_y).real
        )

        def update_horizontal_spatial(rib, fft_boundary, side):

            du_dx = jnp.dot(D_matrix, rib) * jac_sem

            du_dy = (
                jnp.roll(rib, -1, axis=1)
                - jnp.roll(rib, 1, axis=1)
            ) / (2.0 * dx)

            d_dt = -(cx * du_dx + cy * du_dy)

            if side == "top":

                d_dt = d_dt.at[0:2, :].add(
                    penalty
                    * (
                        fft_boundary[-2:, :]
                        - rib[0:2, :]
                    )
                )

            else:

                d_dt = d_dt.at[-2:, :].add(
                    penalty
                    * (
                        fft_boundary[0:2, :]
                        - rib[-2:, :]
                    )
                )

            return d_dt

        d_rib_top_dt = update_horizontal_spatial(
            u_rib_top,
            u_fft,
            "top",
        )

        d_rib_bot_dt = update_horizontal_spatial(
            u_rib_bot,
            u_fft,
            "bot",
        )

        def update_vertical_spatial(rib, fft_boundary, side):

            du_dx = (
                jnp.roll(rib, -1, axis=0)
                - jnp.roll(rib, 1, axis=0)
            ) / (2.0 * dx)

            du_dy = jnp.dot(
                rib,
                D_matrix.T,
            ) * jac_sem            

            d_dt = -(cx * du_dx + cy * du_dy)

            if side == "left":

                d_dt = d_dt.at[:, -2:].add(
                    penalty
                    * (
                        fft_boundary[:, 0:2]
                        - rib[:, -2:]
                    )
                )

            else:

                d_dt = d_dt.at[:, 0:2].add(
                    penalty
                    * (
                        fft_boundary[:, -2:]
                        - rib[:, 0:2]
                    )
                )

            return d_dt

        d_rib_left_dt = update_vertical_spatial(
            u_rib_left,
            u_fft,
            "left",
        )

        d_rib_right_dt = update_vertical_spatial(
            u_rib_right,
            u_fft,
            "right",
        )

        return (
            d_fft_dt,
            d_rib_top_dt,
            d_rib_bot_dt,
            d_rib_left_dt,
            d_rib_right_dt,
        )

    state = (
        u_fft,
        u_rib_top,
        u_rib_bot,
        u_rib_left,
        u_rib_right,
    )

    k1 = rhs(*state)

    def add_k(state, k, factor):
        return tuple(
            s + factor * dk
            for s, dk in zip(state, k)
        )

    k2 = rhs(*add_k(state, k1, 0.5 * dt))
    k3 = rhs(*add_k(state, k2, 0.5 * dt))
    k4 = rhs(*add_k(state, k3, dt))

    return tuple(
        state[i]
        + (dt / 6.0)
        * (
            k1[i]
            + 2.0 * k2[i]
            + 2.0 * k3[i]
            + k4[i]
        )
        for i in range(5)
    )


# ============================================================
# OPTIMIZED SINC HYBRID
# ============================================================

@jax.jit
def run_hybrid_sfx_2d_sinc(
    u_fft,
    u_rib_top,
    u_rib_bot,
    u_rib_left,
    u_rib_right,
    ik_x,
    ik_y,
    D_matrix,
    jac_sem,
    cx,
    cy,
    dt,
    dx
):

    penalty = 1.0 / dt

    N = u_fft.shape[0]

    # --------------------------------------------------------
    # PRECOMPUTE SINC KERNEL
    # --------------------------------------------------------

    sinc_kernel = jnp.sinc(
        jnp.arange(N, dtype=u_fft.dtype)
        - (N / 2.0)
    )

    sinc_kernel_row = sinc_kernel.reshape(1, N)
    sinc_kernel_col = sinc_kernel.reshape(N, 1)

    # --------------------------------------------------------
    # COUPLING OPERATORS
    # --------------------------------------------------------

    def sinc_horizontal(data, rows):

        projected = sinc_kernel_row @ data

        return jnp.broadcast_to(
            projected,
            (rows, data.shape[1]),
        )

    def sinc_vertical(data, cols):

        projected = data @ sinc_kernel_col

        return jnp.broadcast_to(
            projected,
            (data.shape[0], cols),
        )

    # --------------------------------------------------------
    # RHS
    # --------------------------------------------------------

    def rhs(
        u_fft,
        u_rib_top,
        u_rib_bot,
        u_rib_left,
        u_rib_right,
    ):

        u_fft_hat = jnp.fft.fft2(u_fft)

        d_fft_dt = -(
            cx * jnp.fft.ifft2(u_fft_hat * ik_x).real
            + cy * jnp.fft.ifft2(u_fft_hat * ik_y).real
        )

        # ----------------------------------------------------
        # TOP / BOTTOM
        # ----------------------------------------------------

        def update_horizontal_spatial(
            rib,
            fft_boundary,
            side,
        ):

            du_dx = jnp.dot(D_matrix, rib) * jac_sem

            du_dy = (
                jnp.roll(rib, -1, axis=1)
                - jnp.roll(rib, 1, axis=1)
            ) / (2.0 * dx)

            d_dt = -(cx * du_dx + cy * du_dy)

            coupled_fft = sinc_horizontal(
                fft_boundary,
                rib.shape[0],
            )

            rows = min(2, rib.shape[0])

            if side == "top":

                d_dt = d_dt.at[:rows, :].add(
                    penalty
                    * (
                        coupled_fft[:rows, :]
                        - rib[:rows, :]
                    )
                )

            else:

                d_dt = d_dt.at[-rows:, :].add(
                    penalty
                    * (
                        coupled_fft[-rows:, :]
                        - rib[-rows:, :]
                    )
                )

            return d_dt

        d_rib_top_dt = update_horizontal_spatial(
            u_rib_top,
            u_fft,
            "top",
        )

        d_rib_bot_dt = update_horizontal_spatial(
            u_rib_bot,
            u_fft,
            "bot",
        )

        # ----------------------------------------------------
        # LEFT / RIGHT
        # ----------------------------------------------------

        def update_vertical_spatial(
            rib,
            fft_boundary,
            side,
        ):

            du_dx = (
                jnp.roll(rib, -1, axis=0)
                - jnp.roll(rib, 1, axis=0)
            ) / (2.0 * dx)

            du_dy = jnp.dot(
                rib,
                D_matrix.T,
            ) * jac_sem

            d_dt = -(cx * du_dx + cy * du_dy)

            coupled_fft = sinc_vertical(
                fft_boundary,
                rib.shape[1],
            )

            cols = min(2, rib.shape[1])

            if side == "left":

                d_dt = d_dt.at[:, -cols:].add(
                    penalty
                    * (
                        coupled_fft[:, -cols:]
                        - rib[:, -cols:]
                    )
                )

            else:

                d_dt = d_dt.at[:, :cols].add(
                    penalty
                    * (
                        coupled_fft[:, :cols]
                        - rib[:, :cols]
                    )
                )

            return d_dt

        d_rib_left_dt = update_vertical_spatial(
            u_rib_left,
            u_fft,
            "left",
        )

        d_rib_right_dt = update_vertical_spatial(
            u_rib_right,
            u_fft,
            "right",
        )

        return (
            d_fft_dt,
            d_rib_top_dt,
            d_rib_bot_dt,
            d_rib_left_dt,
            d_rib_right_dt,
        )

    state = (
        u_fft,
        u_rib_top,
        u_rib_bot,
        u_rib_left,
        u_rib_right,
    )

    k1 = rhs(*state)

    def add_k(state, k, factor):
        return tuple(
            s + factor * dk
            for s, dk in zip(state, k)
        )

    k2 = rhs(*add_k(state, k1, 0.5 * dt))
    k3 = rhs(*add_k(state, k2, 0.5 * dt))
    k4 = rhs(*add_k(state, k3, dt))

    return tuple(
        state[i]
        + (dt / 6.0)
        * (
            k1[i]
            + 2.0 * k2[i]
            + 2.0 * k3[i]
            + k4[i]
        )
        for i in range(5)
    )
