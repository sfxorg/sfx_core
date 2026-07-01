# test_spectral_edge.py

import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================

N = 256

L = 1.0

dx = L / N

dt = 2e-4

NSTEPS = 2000

c = 1.0

# panel split
MID = N // 2

# ==========================================================
# GRID
# ==========================================================

x = np.linspace(
    0.0,
    L,
    N,
    endpoint=False
)

# ==========================================================
# INITIAL BUBBLE
# ==========================================================

u0 = np.exp(
    -((x - 0.25)**2) / (0.03**2)
)

# ==========================================================
# FFT WAVENUMBERS
# ==========================================================

ik = (
    1j *
    2.0 *
    np.pi *
    np.fft.fftfreq(
        N,
        d=dx
    )
)

# ==========================================================
# REFERENCE SOLVER
# ==========================================================

u_ref = u0.copy()

# ==========================================================
# TWO PANELS
# ==========================================================

uA = u0[:MID].copy()
uB = u0[MID:].copy()

# ==========================================================
# RHS
# ==========================================================

def fft_rhs(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik * uhat
    ).real

    return -c * ux

# ==========================================================
# "FULL SPECTRAL SCHUR"
#
# Interface represented by
# complete edge spectrum.
# (No truncation yet)
# ==========================================================

def spectral_edge_coupling(uA, uB):

    left_edge = np.array([
        uA[-1],
        uA[-2]
    ])

    right_edge = np.array([
        uB[0],
        uB[1]
    ])

    # virtual spectrums
    LA = np.fft.rfft(left_edge)
    RB = np.fft.rfft(right_edge)

    avg = 0.5 * (LA + RB)

    left_new = np.fft.irfft(
        avg,
        n=2
    )

    right_new = np.fft.irfft(
        avg,
        n=2
    )

    uA[-2:] = left_new
    uB[:2] = right_new

    return uA, uB

# ==========================================================
# RK4
# ==========================================================

def rk4(rhs, u):

    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u + dt*k3)

    return u + (
        dt/6.0
    ) * (
        k1
        + 2*k2
        + 2*k3
        + k4
    )

# ==========================================================
# TIMESTEP
# ==========================================================

for step in range(NSTEPS):

    # reference
    u_ref = rk4(
        fft_rhs,
        u_ref
    )

    # panel coupling
    uA, uB = spectral_edge_coupling(
        uA,
        uB
    )

    # evolve separately
    uA = rk4(
        lambda q: fft_rhs(
            np.pad(
                q,
                (0, MID),
                mode="constant"
            )
        )[:MID],
        uA
    )

    uB = rk4(
        lambda q: fft_rhs(
            np.pad(
                q,
                (MID, 0),
                mode="constant"
            )[MID:],
        ),
        uB
    )

# ==========================================================
# RECONSTRUCT
# ==========================================================

u_two = np.concatenate(
    [uA, uB]
)

# ==========================================================
# DIAGNOSTICS
# ==========================================================

err = np.max(
    np.abs(
        u_ref - u_two
    )
)

print()
print("Max error =", err)

mass_ref = np.sum(u_ref)*dx
mass_two = np.sum(u_two)*dx

print("Mass error =", abs(
    mass_ref - mass_two
))

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(10,4))

plt.plot(
    x,
    u_ref,
    label="Reference FFT",
    lw=2
)

plt.plot(
    x,
    u_two,
    "--",
    label="Two Panel Spectral Edge"
)

plt.axvline(
    x[MID],
    color="k",
    ls=":"
)

plt.legend()

plt.title(
    "Bubble Crossing Interface"
)

plt.tight_layout()

plt.show()
