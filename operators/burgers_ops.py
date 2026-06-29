# operators/burgers_ops.py

import jax
import jax.numpy as jnp


# ============================================================
# FFT BURGERS RHS
# ============================================================

@jax.jit
def burgers_fft_rhs(
    u,
    ik_x,
    ik_y,
):
    """
    2D scalar Burgers:

        u_t + d/dx(0.5*u^2)
            + d/dy(0.5*u^2)
          = 0
    """

    flux = 0.5 * u**2

    flux_hat = jnp.fft.fft2(flux)

    dFdx = jnp.fft.ifft2(
        flux_hat * ik_x
    ).real

    dFdy = jnp.fft.ifft2(
        flux_hat * ik_y
    ).real

    return -(dFdx + dFdy)


# ============================================================
# FV BURGERS RHS (1ST ORDER)
# ============================================================

@jax.jit
def burgers_fv_rhs_1st_order(
    u,
    dx,
    dy,
):
    """
    Conservative first-order FV Burgers
    """

    Fx = 0.5 * u**2
    Fy = 0.5 * u**2

    dFdx = (
        Fx
        - jnp.roll(Fx, 1, axis=0)
    ) / dx

    dFdy = (
        Fy
        - jnp.roll(Fy, 1, axis=1)
    ) / dy

    return -(dFdx + dFdy)


# ============================================================
# FV BURGERS RHS (3RD ORDER)
# ============================================================

@jax.jit
def burgers_fv_rhs_3rd_order(
    u,
    dx,
    dy,
):
    """
    Same reconstruction style as your
    fv_rhs_3rd_order().
    """

    uLx = (
        -jnp.roll(u, -1, axis=0)
        + 5.0 * u
        + 2.0 * jnp.roll(u, 1, axis=0)
    ) / 6.0

    uLy = (
        -jnp.roll(u, -1, axis=1)
        + 5.0 * u
        + 2.0 * jnp.roll(u, 1, axis=1)
    ) / 6.0

    Fx = 0.5 * uLx**2
    Fy = 0.5 * uLy**2

    dFdx = (
        Fx
        - jnp.roll(Fx, 1, axis=0)
    ) / dx

    dFdy = (
        Fy
        - jnp.roll(Fy, 1, axis=1)
    ) / dy

    return -(dFdx + dFdy)


# ============================================================
# HYBRID FFT CORE + SEM RIBBONS
# ============================================================

@jax.jit
def burgers_hybrid_standard(
    u_fft,
    u_rib_top,
    u_rib_bot,
    u_rib_left,
    u_rib_right,
    ik_x,
    ik_y,
    D_matrix,
    jac_sem,
    dt,
    dx,
):

    penalty = 1.0 / dt

    def rhs(
        u_fft,
        u_rib_top,
        u_rib_bot,
        u_rib_left,
        u_rib_right,
    ):

        # =====================================================
        # FFT CORE
        # =====================================================

        flux_fft = 0.5 * u_fft**2

        flux_hat = jnp.fft.fft2(flux_fft)

        d_fft_dt = -(
            jnp.fft.ifft2(
                flux_hat * ik_x
            ).real
            +
            jnp.fft.ifft2(
                flux_hat * ik_y
            ).real
        )

        # =====================================================
        # RIBBON OPERATORS
        # =====================================================

        def update_horizontal(
            rib,
            fft_boundary,
            side,
        ):

            Fx = 0.5 * rib**2
            Fy = 0.5 * rib**2

            dFdx = (
                jnp.dot(
                    D_matrix,
                    Fx,
                )
                * jac_sem
            )

            dFdy = (
                jnp.roll(Fy, -1, axis=1)
                - jnp.roll(Fy, 1, axis=1)
            ) / (2.0 * dx)

            d_dt = -(dFdx + dFdy)

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

        def update_vertical(
            rib,
            fft_boundary,
            side,
        ):

            Fx = 0.5 * rib**2
            Fy = 0.5 * rib**2

            dFdx = (
                jnp.roll(Fx, -1, axis=0)
                - jnp.roll(Fx, 1, axis=0)
            ) / (2.0 * dx)

            dFdy = (
                jnp.dot(
                    Fy,
                    D_matrix.T,
                )
                * jac_sem
            )

            d_dt = -(dFdx + dFdy)

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

        # =====================================================
        # COMPUTE RIBBON RHS
        # =====================================================

        d_rib_top_dt = update_horizontal(
            u_rib_top,
            u_fft,
            "top",
        )

        d_rib_bot_dt = update_horizontal(
            u_rib_bot,
            u_fft,
            "bot",
        )

        d_rib_left_dt = update_vertical(
            u_rib_left,
            u_fft,
            "left",
        )

        d_rib_right_dt = update_vertical(
            u_rib_right,
            u_fft,
            "right",
        )

        # =====================================================
        # RHS FEEDBACK
        # =====================================================

        alpha = 0.5

        top_rhs_residual = (
            d_rib_top_dt[0:2, :]
            - d_fft_dt[-2:, :]
        )

        bot_rhs_residual = (
            d_rib_bot_dt[-2:, :]
            - d_fft_dt[0:2, :]
        )

        left_rhs_residual = (
            d_rib_left_dt[:, -2:]
            - d_fft_dt[:, 0:2]
        )

        right_rhs_residual = (
            d_rib_right_dt[:, 0:2]
            - d_fft_dt[:, -2:]
        )

        d_fft_dt = d_fft_dt.at[-2:, :].add(
            alpha * top_rhs_residual
        )

        d_fft_dt = d_fft_dt.at[0:2, :].add(
            alpha * bot_rhs_residual
        )

        d_fft_dt = d_fft_dt.at[:, 0:2].add(
            alpha * left_rhs_residual
        )

        d_fft_dt = d_fft_dt.at[:, -2:].add(
            alpha * right_rhs_residual
        )

        # =====================================================
        # DIAGNOSTICS
        # =====================================================

        #jax.debug.print(
        #    "state_res={}",
        #    jnp.max(
        #        jnp.abs(
        #            u_rib_top[0:2, :]
        #            - u_fft[-2:, :]
        #        )
        #    )
        #)

        #jax.debug.print(
        #    "rhs_res={}",
        #    jnp.max(
        #        jnp.abs(
        #            top_rhs_residual
        #        )
        #    )
        #)

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
