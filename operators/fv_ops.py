# operators/fv_ops.py

import jax
import jax.numpy as jnp


@jax.jit
def fv_rhs_1st_order(
    u,
    cx,
    cy,
    dx,
    dy,
):

    d_flux_dx = (
        u
        - jnp.roll(u, 1, axis=0)
    ) / dx

    d_flux_dy = (
        u
        - jnp.roll(u, 1, axis=1)
    ) / dy

    return -(cx * d_flux_dx + cy * d_flux_dy)


@jax.jit
def fv_rhs_3rd_order(
    u,
    cx,
    cy,
    dx,
    dy,
):

    uLx = (
        -jnp.roll(u, -1, axis=0)
        + 5.0*u
        + 2.0*jnp.roll(u, 1, axis=0)
    ) / 6.0

    uLy = (
        -jnp.roll(u, -1, axis=1)
        + 5.0*u
        + 2.0*jnp.roll(u, 1, axis=1)
    ) / 6.0

    Fx = cx * uLx
    Fy = cy * uLy

    dFdx = (
        Fx
        - jnp.roll(Fx, 1, axis=0)
    ) / dx

    dFdy = (
        Fy
        - jnp.roll(Fy, 1, axis=1)
    ) / dy

    return -(dFdx + dFdy)
