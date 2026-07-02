# 26_SFX_SAT_POD_Schwarz_V26.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Configuration
# ============================================================

N = 256
L = 1.0

MID = N // 2
W = 32

NU = 1e-2

DT = 1e-4
NSTEPS = 1000

SAT_BETA = 0.5

NTRAIN = 100
ENERGY_CAPTURE = 0.999

POD_RANKS = [16, 22]

rng = np.random.default_rng(12345)

# ============================================================
# Grid
# ============================================================

x = np.linspace(
    0.0,
    L,
    N,
    endpoint=False
)

dx = L / N

freq_full = np.fft.fftfreq(
    N,
    d=dx
)

ik_full = (
    1j *
    2.0 *
    np.pi *
    freq_full
)

k2_full = -(
    2.0 *
    np.pi *
    freq_full
) ** 2

freq_half = np.fft.fftfreq(
    MID,
    d=dx
)

ik_half = (
    1j *
    2.0 *
    np.pi *
    freq_half
)

k2_half = -(
    2.0 *
    np.pi *
    freq_half
) ** 2

# ============================================================
# PDE
# ============================================================

def flux(u):
    return 0.5 * u**2


def rhs_full(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik_full * uhat
    ).real

    uxx = np.fft.ifft(
        k2_full * uhat
    ).real

    return (
        -(u * ux)
        + NU * uxx
    )


def rhs_half(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik_half * uhat
    ).real

    uxx = np.fft.ifft(
        k2_half * uhat
    ).real

    return (
        -(u * ux)
        + NU * uxx
    )


def rk4(rhs, u):

    k1 = rhs(u)

    k2 = rhs(
        u + 0.5 * DT * k1
    )

    k3 = rhs(
        u + 0.5 * DT * k2
    )

    k4 = rhs(
        u + DT * k3
    )

    return (
        u
        + DT
        * (
            k1
            + 2*k2
            + 2*k3
            + k4
        ) / 6.0
    )

# ============================================================
# Training waves
# ============================================================

def random_wave():

    u = np.zeros_like(x)

    nmodes = rng.integers(
        3,
        8
    )

    ks = rng.choice(
        np.arange(2, 40),
        size=nmodes,
        replace=False
    )

    for k in ks:

        amp = rng.uniform(
            0.2,
            1.0
        )

        phase = rng.uniform(
            0,
            2*np.pi
        )

        u += (
            amp *
            np.sin(
                2*np.pi*k*x
                + phase
            )
        )

    return u

# ============================================================
# Build POD manifold of FLUX interfaces
# ============================================================

print()
print("Building SAT-FLUX POD basis...")

snapshots = []

for sample in range(NTRAIN):

    u = random_wave()

    for step in range(200):

        u = rk4(
            rhs_full,
            u
        )

        f = flux(u)

        left = f[
            MID-W:MID
        ]

        right = f[
            MID:MID+W
        ]

        snapshots.append(
            np.concatenate(
                [left, right]
            )
        )

X = np.array(
    snapshots
).T

mean_vec = np.mean(
    X,
    axis=1,
    keepdims=True
)

Xc = X - mean_vec

U, S, VT = np.linalg.svd(
    Xc,
    full_matrices=False
)

energy = np.cumsum(
    S**2
)

energy /= energy[-1]

full_rank = (
    np.searchsorted(
        energy,
        ENERGY_CAPTURE
    )
    + 1
)

print(
    "Flux POD Rank =",
    full_rank
)

# ============================================================
# Reference solution
# ============================================================

u0 = (
    np.sin(2*np.pi*12*x)
    +
    0.5*np.sin(2*np.pi*24*x)
    +
    0.25*np.sin(2*np.pi*48*x)
)

u_ref = u0.copy()

for _ in range(NSTEPS):

    u_ref = rk4(
        rhs_full,
        u_ref
    )

# ============================================================
# SAT Schwarz
# ============================================================

def run_case(rank=None):

    uA = u0[:MID].copy()
    uB = u0[MID:].copy()

    basis = None

    if rank is not None:
        basis = U[:, :rank]

    for _ in range(NSTEPS):

        fA = flux(uA[-W:])
        fB = flux(uB[:W])

        state = np.concatenate(
            [fA, fB]
        )

        if rank is not None:

            centered = (
                state.reshape(-1,1)
                - mean_vec
            )

            coeff = (
                basis.T
                @ centered
            )

            state = (
                basis @ coeff
            ).flatten()

            state += mean_vec.flatten()

        fluxA = state[:W]
        fluxB = state[W:]

        # SAT-like interface penalty

        penaltyA = (
            SAT_BETA
            *
            (fluxB - fA)
        )

        penaltyB = (
            SAT_BETA
            *
            (fluxA - fB)
        )

        uA[-W:] += DT * penaltyA
        uB[:W] += DT * penaltyB

        uA = rk4(
            rhs_half,
            uA
        )

        uB = rk4(
            rhs_half,
            uB
        )

    sol = np.concatenate(
        [uA, uB]
    )

    err = (
        np.linalg.norm(
            sol - u_ref
        )
        /
        (
            np.linalg.norm(
                u_ref
            )
            + 1e-14
        )
    )

    return err

# ============================================================
# Run Cases
# ============================================================

results = {}

print()
print("Running FULL SAT")

results["FULL"] = run_case(
    rank=None
)

for r in POD_RANKS:

    print(
        "Running POD",
        r
    )

    results[f"POD{r}"] = (
        run_case(rank=r)
    )

# ============================================================
# Report
# ============================================================

pdf_name = (
    "test_sfx_v26_sat_pod_schwarz.pdf"
)

with PdfPages(pdf_name) as pdf:

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    txt = ""

    txt += (
        "V26 SAT POD SCHWARZ\n\n"
    )

    txt += (
        f"Flux POD Rank = "
        f"{full_rank}\n\n"
    )

    txt += (
        "MODEL      ERROR\n\n"
    )

    for k, v in results.items():

        txt += (
            f"{k:10s}"
            f"{v:.6e}\n"
        )

    txt += "\n"

    txt += (
        "FULL DOF = 64\n"
    )

    txt += (
        "POD22 DOF = 22\n"
    )

    txt += (
        "POD16 DOF = 16\n"
    )

    plt.text(
        0.01,
        0.99,
        txt,
        va="top",
        family="monospace"
    )

    pdf.savefig(fig)
    plt.close(fig)

print()
print(
    "Saved:",
    pdf_name
)
