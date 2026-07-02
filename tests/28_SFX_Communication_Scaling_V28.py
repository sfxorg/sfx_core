import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V28
#
# Communication Scaling Study
#
# FULL
# POD22
# POD16
# ============================================================

N = 256
L = 1.0

MID = N // 2
W = 32

DT = 1e-4
NSTEPS = 1000

NU = 1e-2

NTRAIN = 100
ENERGY_CAPTURE = 0.999

DOUBLE_BYTES = 8

rng = np.random.default_rng(12345)

# ============================================================
# Grid
# ============================================================

x = np.linspace(0.0, L, N, endpoint=False)

dx = L / N

freq_full = np.fft.fftfreq(N, d=dx)

ik_full = 1j * 2.0 * np.pi * freq_full
k2_full = -(2.0 * np.pi * freq_full) ** 2

freq_half = np.fft.fftfreq(MID, d=dx)

ik_half = 1j * 2.0 * np.pi * freq_half
k2_half = -(2.0 * np.pi * freq_half) ** 2

# ============================================================
# Burgers
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

    return -(u * ux) + NU * uxx


def rhs_half(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik_half * uhat
    ).real

    uxx = np.fft.ifft(
        k2_half * uhat
    ).real

    return -(u * ux) + NU * uxx


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
# Training Data
# ============================================================

def random_wave():

    u = np.zeros_like(x)

    nmodes = rng.integers(3, 8)

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
            amp
            * np.sin(
                2*np.pi*k*x
                + phase
            )
        )

    return u


# ============================================================
# Build Flux POD Basis
# ============================================================

print("Building flux POD basis...")

snapshots = []

for sample in range(NTRAIN):

    u = random_wave()

    for step in range(200):

        u = rk4(
            rhs_full,
            u
        )

        f = flux(u)

        snapshots.append(
            np.concatenate(
                [
                    f[MID-W:MID],
                    f[MID:MID+W]
                ]
            )
        )

X = np.array(snapshots).T

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

energy = np.cumsum(S**2)
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
# Test Signal
# ============================================================

u0 = (
    np.sin(2*np.pi*12*x)
    +
    0.5*np.sin(2*np.pi*24*x)
    +
    0.25*np.sin(2*np.pi*48*x)
)

# ============================================================
# Reference
# ============================================================

u_ref = u0.copy()

for _ in range(NSTEPS):
    u_ref = rk4(rhs_full, u_ref)

# ============================================================
# Run Model
# ============================================================

def run_case(rank=None):

    uA = u0[:MID].copy()
    uB = u0[MID:].copy()

    communicated_scalars = 0

    basis = None

    if rank is not None:
        basis = U[:, :rank]

    for step in range(NSTEPS):

        fA = flux(uA[-W:])
        fB = flux(uB[:W])

        state = np.concatenate([fA, fB])

        if rank is None:

            communicated_scalars += len(state)

        else:

            centered = (
                state.reshape(-1, 1)
                - mean_vec
            )

            coeff = (
                basis.T
                @ centered
            )

            communicated_scalars += rank

            state = (
                basis @ coeff
            ).flatten()

            state += mean_vec.flatten()

        left_flux = state[:W]
        right_flux = state[W:]

        beta = 0.5

        uA[-W:] += DT * beta * (
            right_flux - fA
        )

        uB[:W] += DT * beta * (
            left_flux - fB
        )

        uA = rk4(rhs_half, uA)
        uB = rk4(rhs_half, uB)

    sol = np.concatenate([uA, uB])

    error = (
        np.linalg.norm(sol - u_ref)
        /
        (
            np.linalg.norm(u_ref)
            + 1e-14
        )
    )

    bytes_sent = (
        communicated_scalars
        * DOUBLE_BYTES
    )

    return (
        error,
        communicated_scalars,
        bytes_sent
    )

# ============================================================
# Cases
# ============================================================

cases = {
    "FULL": None,
    "POD22": 22,
    "POD16": 16,
}

results = {}

for name, rank in cases.items():

    print("Running", name)

    results[name] = run_case(rank)

# ============================================================
# PDF
# ============================================================

pdf_name = (
    "test_sfx_v28_communication_scaling.pdf"
)

with PdfPages(pdf_name) as pdf:

    fig = plt.figure(figsize=(11,8))

    plt.axis("off")

    txt = ""

    txt += "V28 COMMUNICATION SCALING\n\n"

    txt += (
        "MODEL      "
        "ERROR        "
        "SCALARS        "
        "BYTES\n\n"
    )

    for name, data in results.items():

        err, scalars, bytes_sent = data

        txt += (
            f"{name:8s} "
            f"{err:10.4e} "
            f"{scalars:12d} "
            f"{bytes_sent:12d}\n"
        )

    full_bytes = results["FULL"][2]

    txt += "\nCompression\n\n"

    for name in ["POD22", "POD16"]:

        ratio = (
            full_bytes
            /
            results[name][2]
        )

        txt += (
            f"{name:8s} "
            f"{ratio:.2f}x\n"
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
