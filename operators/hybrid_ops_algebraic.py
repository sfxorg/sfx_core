import jax
import jax.numpy as jnp


# ============================================================
# SPECTRAL RIBBON OPERATORS
# ============================================================

def make_spectral_ribbon_y(N, L):

    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(
        N,
        d=L / N,
    )

    @jax.jit
    def ribbon(u_hat, ys):

        phase = jnp.exp(
            1j * ys[:, None] * ky[None, :]
        )

        trace_hat = jnp.einsum(
            "pm,mn->pn",
            phase,
            u_hat.T,
        )

        return jnp.fft.ifft(
            trace_hat,
            axis=1,
        ).real

    return ribbon


def make_spectral_ribbon_x(N, L):

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(
        N,
        d=L / N,
    )

    @jax.jit
    def ribbon(u_hat, xs):

        phase = jnp.exp(
            1j * xs[:, None] * kx[None, :]
        )

        trace_hat = jnp.einsum(
            "pm,mn->pn",
            phase,
            u_hat,
        )

        rib = jnp.fft.ifft(
            trace_hat,
            axis=1,
        ).real

        return rib.T

    return ribbon


# ============================================================
# HYBRID ALGEBRAIC RHS
# ============================================================

def make_hybrid_fft_rhs_algebraic(
    N,
    L,
    P,
):

    spectral_y = make_spectral_ribbon_y(
        N,
        L,
    )

    spectral_x = make_spectral_ribbon_x(
        N,
        L,
    )

    dx = L / N

    ys_top = L - dx * (
        jnp.arange(P + 1) + 1
    )

    ys_bot = dx * jnp.arange(P + 1)

    xs_left = dx * jnp.arange(P + 1)

    xs_right = L - dx * (
        jnp.arange(P + 1) + 1
    )

    @jax.jit
    def rhs(
        u,
        ik_x,
        ik_y,
        cx,
        cy,
        penalty,
    ):

        # ----------------------------------------------------
        # FFT advection
        # ----------------------------------------------------

        u_hat = jnp.fft.fft2(u)

        rhs_fft = -(
            cx * jnp.fft.ifft2(
                u_hat * ik_x
            ).real
            +
            cy * jnp.fft.ifft2(
                u_hat * ik_y
            ).real
        )

        # ----------------------------------------------------
        # Spectral traces
        # ----------------------------------------------------

        top_trace = spectral_y(
            u_hat,
            ys_top,
        )

        bot_trace = spectral_y(
            u_hat,
            ys_bot,
        )

        left_trace = spectral_x(
            u_hat,
            xs_left,
        )

        right_trace = spectral_x(
            u_hat,
            xs_right,
        )

        # ----------------------------------------------------
        # Residuals
        # ----------------------------------------------------

        top_res = (
            top_trace
            - u[:P + 1, :]
        )

        bot_res = (
            bot_trace
            - u[-(P + 1):, :]
        )

        left_res = (
            left_trace
            - u[:, :P + 1]
        )

        right_res = (
            right_trace
            - u[:, -(P + 1):]
        )

        # ----------------------------------------------------
        # SAT coupling
        # ----------------------------------------------------

        rhs_fft = rhs_fft.at[
            :P + 1,
            :
        ].add(
            penalty * top_res
        )

        rhs_fft = rhs_fft.at[
            -(P + 1):,
            :
        ].add(
            penalty * bot_res
        )

        rhs_fft = rhs_fft.at[
            :,
            :P + 1
        ].add(
            penalty * left_res
        )

        rhs_fft = rhs_fft.at[
            :,
            -(P + 1):
        ].add(
            penalty * right_res
        )

        return rhs_fft

    return rhs
