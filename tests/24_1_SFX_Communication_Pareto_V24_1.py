# 24_1_SFX_Communication_Pareto_V24_1.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Configuration
# ============================================================

N = 256
L = 1.0

NU = 1e-2

DT = 1e-4
NSTEPS = 1000

W = 32

ENERGY_CAPTURE = 0.999

NTRAIN = 100

RANKS_TO_TEST = [
    1,
    2,
    4,
    8,
    12,
    16,
    22,
]

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

    return u + DT * (
        k1
        + 2*k2v
        + 2*k3
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

FULL_DOF = 2 * W

# ============================================================
# Random Training Signal
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
# Build Ensemble POD Basis
# ============================================================

print()
print("Building ensemble POD basis...")

snapshots = []

for sample in range(NTRAIN):

    u = random_wave()

    for step in range(200):

        u = rk4(u)

        if not np.all(
            np.isfinite(u)
        ):
            break

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
    "Full POD Rank =",
    full_rank
)

# ============================================================
# OOD Cases
# ============================================================

rng2 = np.random.default_rng(999)

random_ood = np.zeros_like(x)

for k in [7,13,19,31]:

    random_ood += (
        rng2.uniform(-1,1)
        *
        np.sin(
            2*np.pi*k*x
        )
    )

cases = {

    "Sin12":
        np.sin(
            2*np.pi*12*x
        ),

    "Sin24":
        np.sin(
            2*np.pi*24*x
        ),

    "BroadbandOOD":
        (
            np.sin(2*np.pi*12*x)
            +
            0.5*np.sin(2*np.pi*24*x)
            +
            0.25*np.sin(2*np.pi*48*x)
        ),

    "Gaussian":
        np.exp(
            -((x-0.30)**2)
            /(0.05**2)
        ),

    "RandomFourierOOD":
        random_ood,
}

# ============================================================
# Evaluate
# ============================================================

all_results = []

pdf_name = (
    "test_sfx_v24_1_comm_pareto.pdf"
)

with PdfPages(pdf_name) as pdf:

    for case_name, u0 in cases.items():

        print(
            "Case:",
            case_name
        )

        rank_errors = []

        for rank in RANKS_TO_TEST:

            basis = U[:, :rank]

            u = u0.copy()

            errs = []

            stable = True

            for step in range(NSTEPS):

                u = rk4(u)

                if not np.all(
                    np.isfinite(u)
                ):
                    stable = False
                    break

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

                err = (
                    np.linalg.norm(
                        centered - recon
                    )
                    /
                    (
                        np.linalg.norm(
                            centered
                        )
                        + 1e-14
                    )
                )

                errs.append(err)

            avg_err = (
                np.mean(errs)
                if len(errs)
                else np.nan
            )

            ratio = (
                FULL_DOF / rank
            )

            rank_errors.append(
                (
                    rank,
                    ratio,
                    avg_err
                )
            )

            all_results.append(
                (
                    case_name,
                    rank,
                    ratio,
                    avg_err
                )
            )

        fig, ax = plt.subplots(
            figsize=(7,5)
        )

        ax.loglog(
            [r[1] for r in rank_errors],
            [r[2] for r in rank_errors],
            "o-"
        )

        ax.set_title(
            case_name
        )

        ax.set_xlabel(
            "Compression Ratio"
        )

        ax.set_ylabel(
            "Average Projection Error"
        )

        pdf.savefig(fig)
        plt.close(fig)

    # Summary Page

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    txt = ""

    txt += (
        "V24.1 COMMUNICATION PARETO\n\n"
    )

    txt += (
        f"Full Interface DOF = "
        f"{FULL_DOF}\n"
    )

    txt += (
        f"Full POD Rank = "
        f"{full_rank}\n\n"
    )

    txt += (
        "Case  Rank  Compression  AvgErr\n\n"
    )

    for row in all_results:

        txt += (
            f"{row[0]:18s} "
            f"{row[1]:4d} "
            f"{row[2]:10.2f} "
            f"{row[3]:10.4e}\n"
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
