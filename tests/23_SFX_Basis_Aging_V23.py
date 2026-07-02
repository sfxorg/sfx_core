import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V23 - Basis Aging Study
# ============================================================

N = 256
L = 1.0

NU = 1.0e-2

DT = 1.0e-4

TRAIN_STEPS = 200
TEST_STEPS = 5000

ENERGY_CAPTURE = 0.999

W = 32
NTRAIN = 100

rng = np.random.default_rng(12345)

# ============================================================
# Grid
# ============================================================

x = np.linspace(0, L, N, endpoint=False)

dx = L / N

freq = np.fft.fftfreq(N, d=dx)

ik = 1j * 2.0 * np.pi * freq

k2 = -(2.0 * np.pi * freq) ** 2

# ============================================================
# Burgers
# ============================================================

def rhs(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik * uhat
    ).real

    uxx = np.fft.ifft(
        k2 * uhat
    ).real

    return -(u * ux) + NU * uxx


def rk4(u):

    k1 = rhs(u)

    k2v = rhs(
        u + 0.5 * DT * k1
    )

    k3 = rhs(
        u + 0.5 * DT * k2v
    )

    k4 = rhs(
        u + DT * k3
    )

    return u + DT * (
        k1
        + 2 * k2v
        + 2 * k3
        + k4
    ) / 6.0

# ============================================================
# Interface
# ============================================================

def interface_state(u):

    left = u[
        N//2-W :
        N//2
    ]

    right = u[
        N//2 :
        N//2+W
    ]

    return np.concatenate(
        [left, right]
    )

# ============================================================
# Random Waves
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
            *
            np.sin(
                2*np.pi*k*x
                + phase
            )
        )

    return u

# ============================================================
# Train Ensemble POD
# ============================================================

print()
print("Building ensemble POD basis...")

snapshots = []

for sample in range(NTRAIN):

    u = random_wave()

    for step in range(TRAIN_STEPS):

        u = rk4(u)

        if not np.all(
            np.isfinite(u)
        ):
            break

        if step % 10 == 0:

            snapshots.append(
                interface_state(u)
            )

print(
    "Snapshots:",
    len(snapshots)
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

rank = (
    np.searchsorted(
        energy,
        ENERGY_CAPTURE
    )
    + 1
)

BASIS = U[:, :rank]

print(
    "Frozen POD Rank =",
    rank
)

# ============================================================
# Test Signal
# ============================================================

u = (
    np.sin(2*np.pi*12*x)
    +
    0.5*np.sin(2*np.pi*24*x)
    +
    0.25*np.sin(2*np.pi*48*x)
)

# ============================================================
# Aging Horizons
# ============================================================

HORIZONS = [
    200,
    500,
    1000,
    2000,
    5000,
]

errors = {}
history = []

current_step = 0

for target in HORIZONS:

    while current_step < target:

        u = rk4(u)

        current_step += 1

        if not np.all(
            np.isfinite(u)
        ):
            print(
                "Simulation unstable"
            )
            break

    state = interface_state(u)

    centered = (
        state.reshape(-1,1)
        - mean_vec
    )

    coeff = (
        BASIS.T
        @ centered
    )

    recon = (
        BASIS
        @ coeff
    )

    proj_err = (
        np.linalg.norm(
            centered - recon
        )
        /
        (
            np.linalg.norm(centered)
            + 1e-14
        )
    )

    errors[target] = proj_err

    history.append(
        (target, proj_err)
    )

# ============================================================
# PDF Report
# ============================================================

pdf_name = (
    "test_sfx_v23_basis_aging.pdf"
)

with PdfPages(pdf_name) as pdf:

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.plot(
        [h[0] for h in history],
        [h[1] for h in history],
        marker="o"
    )

    ax.set_title(
        "Basis Aging"
    )

    ax.set_xlabel(
        "Time Horizon"
    )

    ax.set_ylabel(
        "Projection Error"
    )

    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    txt = ""
    txt += "V23 BASIS AGING STUDY\n\n"
    txt += f"Frozen POD Rank = {rank}\n\n"

    txt += "Horizon      ProjectionError\n\n"

    for horizon, err in history:

        txt += (
            f"{horizon:6d}      "
            f"{err:.6e}\n"
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
print("Saved:", pdf_name)
