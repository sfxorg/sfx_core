import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V29
# SAT-POD Schwarz Scaling Study
# ============================================================

N = 256
L = 1.0

DT = 1.0e-4
NSTEPS = 1000

NU = 1.0e-2

W = 8

NTRAIN = 100
ENERGY_CAPTURE = 0.999

DOUBLE_BYTES = 8

PANEL_COUNTS = [
    2,
    4,
    8,
    16,
    32,
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

def flux(u):
    return 0.5 * u**2


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

    return (
        u
        + DT
        * (
            k1 +
            2*k2v +
            2*k3 +
            k4
        ) / 6.0
    )

# ============================================================
# Training Data
# ============================================================

def random_wave():

    u = np.zeros(N)

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
# Build Flux POD Basis
# ============================================================

print("Building flux POD basis...")

snapshots = []

MID = N // 2

for sample in range(NTRAIN):

    u = random_wave()

    for _ in range(200):

        u = rk4(u)

        f = flux(u)

        state = np.concatenate(
            [
                f[MID-W:MID],
                f[MID:MID+W]
            ]
        )

        snapshots.append(state)

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
    "Flux POD rank =",
    full_rank
)

# ============================================================
# Reference
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
    u_ref = rk4(u_ref)

# ============================================================
# Scaling Experiment
# ============================================================

def run_case(num_panels, rank):

    panel_size = N // num_panels

    panels = []

    for i in range(num_panels):

        panels.append(
            u0[
                i*panel_size:
                (i+1)*panel_size
            ].copy()
        )

    communicated = 0

    basis = None

    if rank is not None:
        basis = U[:, :rank]

    for step in range(NSTEPS):

        for p in range(num_panels - 1):

            left = panels[p]
            right = panels[p+1]

            flux_left = flux(
                left[-W:]
            )

            flux_right = flux(
                right[:W]
            )

            state = np.concatenate(
                [
                    flux_left,
                    flux_right
                ]
            )

            if rank is None:

                communicated += len(state)

            else:

                centered = (
                    state.reshape(-1,1)
                    - mean_vec
                )

                coeff = (
                    basis.T
                    @ centered
                )

                communicated += rank

                state = (
                    basis @ coeff
                ).flatten()

                state += mean_vec.flatten()

            left_flux = state[:W]
            right_flux = state[W:]

            beta = 0.5

            left[-W:] += (
                DT
                * beta
                * (
                    right_flux
                    - flux_left
                )
            )

            right[:W] += (
                DT
                * beta
                * (
                    left_flux
                    - flux_right
                )
            )

        panels = [
            rk4(p)
            for p in panels
        ]

    sol = np.concatenate(
        panels
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

    bytes_sent = (
        communicated
        * DOUBLE_BYTES
    )

    return err, bytes_sent

# ============================================================
# Cases
# ============================================================

results = []

for npanels in PANEL_COUNTS:

    print()
    print(
        f"Panels={npanels}"
    )

    err_full, bytes_full = (
        run_case(
            npanels,
            None
        )
    )

    err22, bytes22 = (
        run_case(
            npanels,
            22
        )
    )

    err16, bytes16 = (
        run_case(
            npanels,
            16
        )
    )

    results.append(
        (
            npanels,
            err_full,
            err22,
            err16,
            bytes_full,
            bytes22,
            bytes16
        )
    )

# ============================================================
# Report
# ============================================================

pdf_name = (
    "test_sfx_v29_scaling.pdf"
)

with PdfPages(pdf_name) as pdf:

    # Error plot

    fig, ax = plt.subplots(
        figsize=(7,5)
    )

    ax.semilogy(
        PANEL_COUNTS,
        [r[1] for r in results],
        "o-",
        label="FULL"
    )

    ax.semilogy(
        PANEL_COUNTS,
        [r[2] for r in results],
        "o-",
        label="POD22"
    )

    ax.semilogy(
        PANEL_COUNTS,
        [r[3] for r in results],
        "o-",
        label="POD16"
    )

    ax.set_xlabel(
        "Panel Count"
    )

    ax.set_ylabel(
        "Relative Error"
    )

    ax.set_title(
        "Scaling Error"
    )

    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

    # Summary page

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    txt = ""

    txt += (
        "V29 SCALING STUDY\n\n"
    )

    txt += (
        "Panels "
        "ErrFULL "
        "Err22 "
        "Err16 "
        "BytesFULL "
        "Bytes22 "
        "Bytes16\n\n"
    )

    for r in results:

        txt += (
            f"{r[0]:5d} "
            f"{r[1]:.3e} "
            f"{r[2]:.3e} "
            f"{r[3]:.3e} "
            f"{r[4]:8d} "
            f"{r[5]:8d} "
            f"{r[6]:8d}\n"
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
