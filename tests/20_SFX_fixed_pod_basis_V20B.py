# 20_SFX_fixed_pod_basis_V20B.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Configuration
# ============================================================

N = 256
L = 1.0

TRAIN_AMP = 0.5

TEST_AMPS = [
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
]

DT = 1.0e-4
NSTEPS = 1000

NU = 1.0e-2

ENERGY_CAPTURE = 0.999

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

k2 = -(
    2.0 * np.pi * freq
) ** 2

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
        + 2 * k2v
        + 2 * k3
        + k4
    ) / 6.0


# ============================================================
# Interface
# ============================================================

W = 32


def interface_state(u):

    left = u[
        N // 2 - W:
        N // 2
    ]

    right = u[
        N // 2:
        N // 2 + W
    ]

    return np.concatenate(
        [left, right]
    )


# ============================================================
# Train POD
# ============================================================

print()
print("================================")
print("TRAINING POD")
print("AMP =", TRAIN_AMP)
print("================================")

u = (
    TRAIN_AMP
    * np.sin(2*np.pi*8*x)
    +
    0.5*TRAIN_AMP
    * np.sin(2*np.pi*16*x)
)

snapshots = []

for step in range(NSTEPS):

    u = rk4(u)

    if not np.all(
        np.isfinite(u)
    ):
        raise RuntimeError(
            "Training run unstable"
        )

    if step % 5 == 0:
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

rank = (
    np.searchsorted(
        energy,
        ENERGY_CAPTURE
    )
    + 1
)

BASIS = U[:, :rank]

print()
print("Frozen POD Rank =", rank)

# ============================================================
# Testing
# ============================================================

pdf_name = (
    "test_sfx_v20b_fixed_pod_basis.pdf"
)

results = []

with PdfPages(pdf_name) as pdf:

    for AMP in TEST_AMPS:

        print()
        print(
            f"Testing AMP={AMP}"
        )

        u = (
            AMP
            * np.sin(
                2*np.pi*8*x
            )
            +
            0.5*AMP
            * np.sin(
                2*np.pi*16*x
            )
        )

        proj_errors = []

        status = "OK"

        for step in range(NSTEPS):

            u = rk4(u)

            if not np.all(
                np.isfinite(u)
            ):
                status = "FAILED"
                break

            state = (
                interface_state(u)
            )

            centered = (
                state.reshape(-1, 1)
                - mean_vec
            )

            coeff = (
                BASIS.T
                @ centered
            )

            recon = (
                BASIS @ coeff
            )

            proj_err = (
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

            proj_errors.append(
                proj_err
            )

        if len(proj_errors) > 0:

            avg_err = float(
                np.mean(
                    proj_errors
                )
            )

            max_err = float(
                np.max(
                    proj_errors
                )
            )

        else:

            avg_err = np.nan
            max_err = np.nan

        results.append(
            (
                AMP,
                status,
                avg_err,
                max_err,
            )
        )

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        if len(proj_errors) > 0:

            ax.plot(
                proj_errors
            )

        ax.set_title(
            f"AMP={AMP} "
            f"STATUS={status}"
        )

        ax.set_ylabel(
            "Projection Error"
        )

        ax.set_xlabel(
            "Step"
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
        "V20B FIXED POD BASIS\n\n"
    )

    txt += (
        f"Training AMP = "
        f"{TRAIN_AMP}\n"
    )

    txt += (
        f"Viscosity = "
        f"{NU}\n"
    )

    txt += (
        f"Frozen POD Rank = "
        f"{rank}\n\n"
    )

    txt += (
        "AMP   STATUS    "
        "AVG_PROJ_ERR    "
        "MAX_PROJ_ERR\n\n"
    )

    for r in results:

        txt += (
            f"{r[0]:4.1f}  "
            f"{r[1]:8s}  "
            f"{r[2]:12.4e}  "
            f"{r[3]:12.4e}\n"
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
