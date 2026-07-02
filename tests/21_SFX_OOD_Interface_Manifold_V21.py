import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Configuration
# ============================================================

N = 256
L = 1.0

TRAIN_AMP = 0.5

DT = 1e-4
NSTEPS = 1000

NU = 1e-2

ENERGY_CAPTURE = 0.999

W = 32

# ============================================================
# Grid
# ============================================================

x = np.linspace(
    0.0,
    L,
    N,
    endpoint=False,
)

dx = L / N

freq = np.fft.fftfreq(
    N,
    d=dx,
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
# TRAINING WAVE FAMILY
# ============================================================

train_u = (
    TRAIN_AMP
    * np.sin(2*np.pi*8*x)
    +
    0.5*TRAIN_AMP
    * np.sin(2*np.pi*16*x)
)

print()
print("Training POD basis...")

snapshots = []

u = train_u.copy()

for step in range(NSTEPS):

    u = rk4(u)

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

print("Frozen POD rank =", rank)

# ============================================================
# OOD TEST CASES
# ============================================================

rng = np.random.default_rng(1234)

random_signal = np.zeros_like(x)

for k in [5, 11, 17, 29]:

    random_signal += (
        rng.uniform(-1, 1)
        * np.sin(
            2*np.pi*k*x
        )
    )

cases = {

    "TrainFamily":
        np.sin(
            2*np.pi*8*x
        )
        +
        0.5*np.sin(
            2*np.pi*16*x
        ),

    "Sin12":
        np.sin(
            2*np.pi*12*x
        ),

    "Sin24":
        np.sin(
            2*np.pi*24*x
        ),

    "BroadbandOOD":
        np.sin(
            2*np.pi*12*x
        )
        +
        0.5*np.sin(
            2*np.pi*24*x
        )
        +
        0.25*np.sin(
            2*np.pi*48*x
        ),

    "Gaussian":
        np.exp(
            -((x-0.30)**2)
            /(0.05**2)
        ),

    "RandomFourier":
        random_signal,
}

# ============================================================
# Test
# ============================================================

results = []

pdf_name = (
    "test_sfx_v21_OOD_interface_manifold.pdf"
)

with PdfPages(pdf_name) as pdf:

    for name, u0 in cases.items():

        print(
            "Testing:",
            name
        )

        u = u0.copy()

        proj_errors = []

        stable = True

        for step in range(NSTEPS):

            u = rk4(u)

            if not np.all(
                np.isfinite(u)
            ):
                stable = False
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

            proj_errors.append(
                err
            )

        avg_err = (
            np.mean(proj_errors)
            if len(proj_errors)
            else np.nan
        )

        max_err = (
            np.max(proj_errors)
            if len(proj_errors)
            else np.nan
        )

        results.append(
            (
                name,
                stable,
                avg_err,
                max_err,
            )
        )

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        if len(proj_errors):
            ax.plot(
                proj_errors
            )

        ax.set_title(
            f"{name}"
        )

        ax.set_ylabel(
            "Projection Error"
        )

        ax.set_xlabel(
            "Step"
        )

        pdf.savefig(fig)
        plt.close(fig)

    # Summary

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    txt = ""

    txt += (
        "V21 OOD INTERFACE MANIFOLD\n\n"
    )

    txt += (
        f"Train AMP={TRAIN_AMP}\n"
    )

    txt += (
        f"Frozen POD Rank={rank}\n\n"
    )

    txt += (
        "Case                "
        "Stable     "
        "AvgErr      "
        "MaxErr\n\n"
    )

    for r in results:

        txt += (
            f"{r[0]:18s} "
            f"{str(r[1]):8s} "
            f"{r[2]:10.4e} "
            f"{r[3]:10.4e}\n"
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
