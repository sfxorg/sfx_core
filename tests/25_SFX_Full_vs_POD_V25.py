import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V25
#
# Full Interface
# vs
# POD-22
# vs
# POD-16
# ============================================================

N = 256
L = 1.0

NU = 1e-2

DT = 1e-4
NSTEPS = 1000

W = 32
NTRAIN = 100

ENERGY_CAPTURE = 0.999

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

freq = np.fft.fftfreq(
    N,
    d=dx
)

ik = 1j * 2.0 * np.pi * freq

k2 = -(2*np.pi*freq)**2

# ============================================================
# PDE
# ============================================================

def rhs(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik * uhat
    ).real

    uxx = np.fft.ifft(
        k2 * uhat
    ).real

    return (
        -(u * ux)
        + NU * uxx
    )

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

    return (
        u
        + DT
        * (
            k1
            + 2*k2v
            + 2*k3
            + k4
        )
        / 6.0
    )

# ============================================================
# Interface
# ============================================================

def interface_state(u):

    left = u[
        N//2-W:
        N//2
    ]

    right = u[
        N//2:
        N//2+W
    ]

    return np.concatenate(
        [left, right]
    )

# ============================================================
# Random Wave Training
# ============================================================

def random_wave():

    u = np.zeros_like(x)

    nmodes = rng.integers(
        3,
        8
    )

    ks = rng.choice(
        np.arange(2,40),
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
# Ensemble POD Basis
# ============================================================

print()
print("Building ensemble POD basis...")

snapshots = []

for sample in range(NTRAIN):

    u = random_wave()

    for step in range(200):

        u = rk4(u)

        if step % 10 == 0:

            snapshots.append(
                interface_state(u)
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
    "Full POD Rank =",
    full_rank
)

# ============================================================
# OOD Test Signal
# ============================================================

u0 = (
    np.sin(2*np.pi*12*x)
    +
    0.5*np.sin(2*np.pi*24*x)
    +
    0.25*np.sin(2*np.pi*48*x)
)

# ============================================================
# Reference Solve
# ============================================================

print("Reference solve...")

u_ref = u0.copy()

for step in range(NSTEPS):
    u_ref = rk4(u_ref)

# ============================================================
# Evaluate Rank
# ============================================================

def evaluate_rank(rank):

    basis = U[:, :rank]

    u = u0.copy()

    err_hist = []

    for step in range(NSTEPS):

        u = rk4(u)

        state = interface_state(u)

        centered = (
            state.reshape(-1,1)
            - mean_vec
        )

        coeff = (
            basis.T
            @ centered
        )

        recon = (
            basis
            @ coeff
        )

        recon_state = (
            recon.flatten()
            + mean_vec.flatten()
        )

        # emulate compressed interface

        u2 = u.copy()

        u2[
            N//2-W:
            N//2+W
        ] = recon_state

        l2err = (
            np.linalg.norm(
                u2 - u_ref
            )
            /
            (
                np.linalg.norm(
                    u_ref
                )
                + 1e-14
            )
        )

        err_hist.append(
            l2err
        )

    return (
        err_hist,
        err_hist[-1]
    )

# ============================================================
# Cases
# ============================================================

cases = {
    "POD16":16,
    "POD22":22,
}

results = {}

pdf_name = (
    "test_sfx_v25_full_vs_pod.pdf"
)

with PdfPages(pdf_name) as pdf:

    for name, rank in cases.items():

        print(
            "Testing",
            name
        )

        hist, final_err = (
            evaluate_rank(rank)
        )

        results[name] = (
            rank,
            final_err
        )

        fig, ax = plt.subplots(
            figsize=(7,4)
        )

        ax.plot(hist)

        ax.set_title(
            f"{name} "
            f"(Rank={rank})"
        )

        ax.set_ylabel(
            "Relative L2 Error"
        )

        ax.set_xlabel(
            "Step"
        )

        pdf.savefig(fig)
        plt.close(fig)

    # summary

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    txt = ""

    txt += (
        "V25 FULL VS POD\n\n"
    )

    txt += (
        f"Full Interface DOF = "
        f"{2*W}\n\n"
    )

    txt += (
        "Model     "
        "Rank     "
        "FinalError\n\n"
    )

    txt += (
        f"Full      "
        f"{2*W:4d}     "
        f"0.0\n"
    )

    for name in results:

        rank, ferr = (
            results[name]
        )

        txt += (
            f"{name:8s} "
            f"{rank:4d} "
            f"{ferr:.4e}\n"
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
