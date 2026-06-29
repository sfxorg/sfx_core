# test_two_panel_sinc_transport.py

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

Nx_panel = 64
Ny = 128

P = 8

cx = 1.0
cy = 0.0

tfinal = 5.0
dt = 5.0e-4

tau = 1.0 / dt

# ============================================================
# GRIDS
# ============================================================

Lx = 1.0
Ly = 1.0

dx_panel = (Lx / 2.0) / Nx_panel
dy = Ly / Ny

xA = jnp.linspace(
    0.0,
    0.5,
    Nx_panel,
    endpoint=False,
)

xB = jnp.linspace(
    0.5,
    1.0,
    Nx_panel,
    endpoint=False,
)

y = jnp.linspace(
    0.0,
    1.0,
    Ny,
    endpoint=False,
)

XA, YA = jnp.meshgrid(
    xA,
    y,
    indexing="ij",
)

XB, YB = jnp.meshgrid(
    xB,
    y,
    indexing="ij",
)

# ============================================================
# WAVENUMBERS
# ============================================================

kx = 2.0 * jnp.pi * jnp.fft.fftfreq(
    Nx_panel,
    d=dx_panel,
)

ky = 2.0 * jnp.pi * jnp.fft.fftfreq(
    Ny,
    d=dy,
)

ik_x = 1j * kx[:, None]
ik_y = 1j * ky[None, :]

# ============================================================
# SEM OPERATOR
# ============================================================

D = jnp.zeros((P, P))

for i in range(1, P - 1):

    D = D.at[i, i - 1].set(-0.5)
    D = D.at[i, i + 1].set(+0.5)

D = D.at[0, 0].set(-1.0)
D = D.at[0, 1].set(+1.0)

D = D.at[-1, -2].set(-1.0)
D = D.at[-1, -1].set(+1.0)

jac_sem = 1.0

# ============================================================
# INITIAL CONDITION
# ============================================================

uA0 = jnp.exp(
    -200.0
    * (
        (XA - 0.45) ** 2
        + (YA - 0.50) ** 2
    )
)

uB0 = jnp.zeros_like(XB)

uR0 = jnp.zeros(
    (
        P,
        Ny,
    )
)

# ============================================================
# FFT RHS
# ============================================================

@jax.jit
def fft_rhs(u):

    uhat = jnp.fft.fft2(u)

    ux = jnp.fft.ifft2(
        uhat * ik_x
    ).real

    uy = jnp.fft.ifft2(
        uhat * ik_y
    ).real

    return -(cx * ux + cy * uy)

# ============================================================
# SINC FFT -> RIBBON
# ============================================================

@jax.jit
def sinc_project_fft_to_ribbon(
    u_fft,
    P,
):

    N = u_fft.shape[0]

    x_fft = jnp.arange(
        N,
        dtype=u_fft.dtype,
    )

    x_rib = jnp.linspace(
        N - P,
        N - 1,
        P,
        dtype=u_fft.dtype,
    )

    S = jnp.sinc(
        x_rib[:, None]
        - x_fft[None, :]
    )

    return S @ u_fft


# ============================================================
# SINC RIBBON -> FFT
# ============================================================

@jax.jit
def sinc_project_ribbon_to_fft(
    rib,
    width,
):

    P = rib.shape[0]

    x_rib = jnp.arange(
        P,
        dtype=rib.dtype,
    )

    x_fft = jnp.linspace(
        0,
        P - 1,
        width,
        dtype=rib.dtype,
    )

    S = jnp.sinc(
        x_fft[:, None]
        - x_rib[None, :]
    )

    return S @ rib

# ============================================================
# RIBBON RHS
# ============================================================

@jax.jit
def ribbon_rhs(
    rib,
    uA,
):

    du_dx = (
        D @ rib
    ) * jac_sem

    du_dy = (
        jnp.roll(rib, -1, axis=1)
        - jnp.roll(rib, 1, axis=1)
    ) / (2.0 * dy)

    drib = -(cx * du_dx + cy * du_dy)

    incoming = sinc_project_fft_to_ribbon(
        uA,
        rib.shape[0],
    )

    drib = drib + tau * (
        incoming
        - rib
    )

    return drib

# ============================================================
# COUPLED RHS
# ============================================================

@jax.jit
def rhs(
    uA,
    uB,
    rib,
):

    dA = fft_rhs(uA)

    dB = fft_rhs(uB)

    dR = ribbon_rhs(
        rib,
        uA,
    )

    width = 8

    incoming = sinc_project_ribbon_to_fft(
        rib,
        width,
    )

    taper = jnp.exp(
        -jnp.arange(
            width,
            dtype=uB.dtype,
        ) / 2.0
    )[:, None]

    dB = dB.at[:width, :].add(
        tau
        * taper
        * (
            incoming
            - uB[:width, :]
        )
    )

    return (
        dA,
        dB,
        dR,
    )

# ============================================================
# RK4
# ============================================================

@jax.jit
def rk4_step(
    uA,
    uB,
    rib,
):

    state = (
        uA,
        uB,
        rib,
    )

    k1 = rhs(*state)

    def add(s, k, a):
        return tuple(
            si + a * dt * ki
            for si, ki in zip(s, k)
        )

    k2 = rhs(
        *add(state, k1, 0.5)
    )

    k3 = rhs(
        *add(state, k2, 0.5)
    )

    k4 = rhs(
        *add(state, k3, 1.0)
    )

    return tuple(
        s
        + dt / 6.0
        * (
            k1i
            + 2.0 * k2i
            + 2.0 * k3i
            + k4i
        )
        for s, k1i, k2i, k3i, k4i
        in zip(
            state,
            k1,
            k2,
            k3,
            k4,
        )
    )

# ============================================================
# RUN
# ============================================================

uA = uA0
uB = uB0
rib = uR0

mass0 = float(
    jnp.sum(uA)
    + jnp.sum(uB)
    + jnp.sum(rib)
)

nsteps = int(
    tfinal / dt
)

t0 = time.time()

for step in range(nsteps):

    uA, uB, rib = rk4_step(
        uA,
        uB,
        rib,
    )

    if step % 1000 == 0:

        print(
            f"{step:6d} "
            f"Amax={float(jnp.max(uA)):.3e} "
            f"Rmax={float(jnp.max(rib)):.3e} "
            f"Bmax={float(jnp.max(uB)):.3e}"
        )

runtime = time.time() - t0

# ============================================================
# DIAGNOSTICS
# ============================================================

jump = float(
    jnp.max(
        jnp.abs(
            uA[-1, :]
            - uB[0, :]
        )
    )
)

mass_final = float(
    jnp.sum(uA)
    + jnp.sum(uB)
    + jnp.sum(rib)
)

print()
print("=" * 60)
print("SINC TRANSPORT TEST")
print("=" * 60)
print(f"Runtime        : {runtime:.3f} s")
print(f"Interface jump : {jump:.3e}")
print(f"Mass error     : {abs(mass_final-mass0):.3e}")
print("=" * 60)

# ============================================================
# VISUALIZATION
# ============================================================

u_full = jnp.concatenate(
    [
        uA,
        uB,
    ],
    axis=0,
)

plt.figure(
    figsize=(12, 5)
)

plt.imshow(
    u_full.T,
    origin="lower",
    aspect="auto",
)

plt.axvline(
    Nx_panel,
    color="white",
    linestyle="--",
    linewidth=2,
)

plt.colorbar()

plt.title(
    "Sinc Projected Transport: FFT A -> Ribbon -> FFT B"
)

plt.tight_layout()

plt.savefig(
    "two_panel_sinc_transport.png",
    dpi=200,
)

print(
    "Saved: two_panel_sinc_transport.png"
)
