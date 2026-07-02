import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V24 Communication Cost Analysis
# ============================================================

N = 256
L = 1.0

NU = 1e-2

DT = 1e-4
NSTEPS = 1000

W = 32

ENERGY_CAPTURE = 0.999
NTRAIN = 100

rng = np.random.default_rng(12345)

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
        + 2 * k2v
        + 2 * k3
        + k4
    ) / 6.0

# ============================================================
# Interface
# ============================================================

def interface_state(u):

    left = u[
        N//2 - W:
        N//2
    ]

    right = u[
        N//2:
        N//2 + W
    ]

    return np.concatenate(
        [left, right]
    )

FULL_INTERFACE_DOF = 2 * W

# ============================================================
# Random Training Wave
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
            amp
            * np.sin(
                2*np.pi*k*x
                + phase
            )
        )

    return u

# ============================================================
# Build Ensemble POD Basis
# ============================================================

print()
print("Building ensemble POD manifold...")

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

compression_ratio = (
    FULL_INTERFACE_DOF / rank
)

print()
print("Frozen POD Rank =", rank)
print(
    "Compression Ratio =",
    compression_ratio
)

# ============================================================
# OOD Test Cases
# ============================================================

rng2 = np.random.default_rng(999)

random_ood = np.zeros_like(x)

for k in [7, 13, 19, 31]:

    random_ood += (
        rng2.uniform(-1, 1)
        * np.sin(2*np.pi*k*x)
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
            -((x-0.3)**2)
            /(0.05**2)
        ),

    "RandomFourierOOD":
        random_ood,
}

# ============================================================
# Evaluate Communication Cost
# ============================================================

results = []

pdf_name = (
    "test_sfx_v24_communication_cost.pdf"
)

with PdfPages(pdf_name) as pdf:

    for name, u0 in cases.items():

        print(
            "Testing:",
            name
        )

        u = u0.copy()

        errs = []

        for step in range(NSTEPS):

            u = rk4(u)

            state = interface_state(u)

            centered = (
                state.reshape(-1,1)
                - mean_vec
            )

            coeff = (
                BASIS.T @ centered
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

            errs.append(err)

        avg_err = np.mean(errs)
        max_err = np.max(errs)

        results.append(
            (
                name,
                avg_err,
                max_err
            )
        )

        fig, ax = plt.subplots(
            figsize=(8,4)
        )

        ax.plot(errs)

        ax.set_title(name)

        ax.set_ylabel(
            "Projection Error"
        )

        ax.set_xlabel(
            "Step"
        )

        pdf.savefig(fig)
        plt.close(fig)

    # ========================================================
    # Summary
    # ========================================================

    fig = plt.figure(figsize=(11, 8))

    plt.axis("off")

    txt = ""

    txt += "V24 COMMUNICATION COST\n\n"

    txt += (
        f"Full Interface DOF = "
        f"{FULL_INTERFACE_DOF}\n"
    )

    txt += (
        f"POD Rank = "
        f"{rank}\n"
    )

    txt += (
        f"Compression Ratio = "
        f"{compression_ratio:.2f}x\n\n"
    )

    txt += (
        "CASE                    "
        "AVG_ERR        "
        "MAX_ERR\n\n"
    )

    for row in results:

        name = row[0]
        avg_err = row[1]
        max_err = row[2]

        txt += (
            f"{name:20s} "
            f"{avg_err:12.4e} "
            f"{max_err:12.4e}\n"
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
