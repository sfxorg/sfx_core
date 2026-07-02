import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V20 Fixed POD Basis Generalization Study
#
# Train POD at AMP=0.5
# Freeze basis
# Reuse for AMP=1,2,5,10
# ============================================================

N = 256
L = 1.0

TRAIN_AMP = 0.5

TEST_AMPS = [
    1.0,
    2.0,
    5.0,
    10.0,
]

DT = 2e-4
NSTEPS = 1000

ENERGY_CAPTURE = 0.999

x = np.linspace(
    0.0,
    L,
    N,
    endpoint=False,
)

dx = L / N

ik = 1j * 2.0 * np.pi * np.fft.fftfreq(
    N,
    d=dx,
)

# ============================================================
# Burgers
# ============================================================

NU = 1e-3


def rhs(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik * uhat
    ).real

    uxx = np.fft.ifft(
        -(2.0 * np.pi *
          np.fft.fftfreq(N, d=dx))**2
        * uhat
    ).real

    return -(u * ux) + NU * uxx


def rk4(u):

    k1 = rhs(u)
    k2 = rhs(u + 0.5 * DT * k1)
    k3 = rhs(u + 0.5 * DT * k2)
    k4 = rhs(u + DT * k3)

    return u + DT * (
        k1 + 2 * k2 + 2 * k3 + k4
    ) / 6.0


# ============================================================
# Interface extractor
# ============================================================

W = 32


def interface_state(u):

    left = u[N // 2 - W:N // 2]
    right = u[N // 2:N // 2 + W]

    return np.concatenate(
        [left, right]
    )


# ============================================================
# TRAIN POD BASIS
# ============================================================

print()
print("===================================")
print("TRAIN POD BASIS")
print("AMP =", TRAIN_AMP)
print("===================================")

u = (
    TRAIN_AMP * np.sin(2 * np.pi * 8 * x)
    + 0.5 * TRAIN_AMP
    * np.sin(2 * np.pi * 16 * x)
)

snapshots = []

for step in range(NSTEPS):

    u = rk4(u)

    if step % 5 == 0:
        snapshots.append(
            interface_state(u)
        )

X = np.array(snapshots).T

mean_vec = np.mean(
    X,
    axis=1,
    keepdims=True,
)

Xc = X - mean_vec

U, S, VT = np.linalg.svd(
    Xc,
    full_matrices=False,
)

energy = np.cumsum(S**2)
energy /= energy[-1]

rank = (
    np.searchsorted(
        energy,
        ENERGY_CAPTURE,
    )
    + 1
)

BASIS = U[:, :rank]

print()
print("Frozen POD rank =", rank)

# ============================================================
# TEST GENERALIZATION
# ============================================================

results = []

pdf_name = (
    "test_sfx_v20_fixed_pod_basis.pdf"
)

with PdfPages(pdf_name) as pdf:

    for AMP in TEST_AMPS:

        print()
        print("Testing AMP =", AMP)

        u = (
            AMP * np.sin(
                2 * np.pi * 8 * x
            )
            + 0.5 * AMP * np.sin(
                2 * np.pi * 16 * x
            )
        )

        proj_errors = []

        snapshots = []

        for step in range(NSTEPS):

            u = rk4(u)

            state = interface_state(u)

            centered = (
                state.reshape(-1, 1)
                - mean_vec
            )

            coeff = (
                BASIS.T @ centered
            )

            recon = (
                BASIS @ coeff
            )

            relerr = (
                np.linalg.norm(
                    centered - recon
                )
                /
                (
                    np.linalg.norm(centered)
                    + 1e-14
                )
            )

            proj_errors.append(
                relerr
            )

            if step % 50 == 0:
                snapshots.append(
                    state.copy()
                )

        avg_proj = np.mean(
            proj_errors
        )

        max_proj = np.max(
            proj_errors
        )

        results.append(
            (
                AMP,
                avg_proj,
                max_proj,
            )
        )

        fig, ax = plt.subplots(
            1,
            2,
            figsize=(11, 4),
        )

        ax[0].plot(
            proj_errors
        )

        ax[0].set_title(
            f"AMP={AMP} Projection Error"
        )

        for s in snapshots[::4]:
            ax[1].plot(
                s,
                alpha=0.4,
            )

        ax[1].set_title(
            f"AMP={AMP} Interface States"
        )

        pdf.savefig(fig)
        plt.close(fig)

    # Summary page

    fig = plt.figure(
        figsize=(11, 8)
    )

    plt.axis("off")

    txt = ""

    txt += (
        "V20 FIXED POD BASIS\n\n"
    )

    txt += (
        f"Training AMP = "
        f"{TRAIN_AMP}\n"
    )

    txt += (
        f"POD Rank = "
        f"{rank}\n\n"
    )

    txt += (
        "AMP      AvgProjErr"
        "      MaxProjErr\n\n"
    )

    for AMP, avgp, maxp in results:

        txt += (
            f"{AMP:5.1f}    "
            f"{avgp:10.4e}    "
            f"{maxp:10.4e}\n"
        )

    plt.text(
        0.01,
        0.99,
        txt,
        va="top",
        family="monospace",
    )

    pdf.savefig(fig)
    plt.close(fig)

print()
print("Saved:", pdf_name)
