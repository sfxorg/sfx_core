import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# V29.1
#
# SAT-POD Schwarz Scaling Study
#
# Fixed version:
# panel-local spectral operators
#
# ============================================================

N = 256
L = 1.0

DT = 1.0e-4
NSTEPS = 1000

NU = 1.0e-2

W = 4

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
# Global Grid
# ============================================================

x = np.linspace(
    0.0,
    L,
    N,
    endpoint=False,
)

dx = L / N

# ============================================================
# Flux
# ============================================================

def flux(u):
    return 0.5 * u**2

# ============================================================
# Local spectral Burgers operator
# ============================================================

def rhs_panel(u):

    n = len(u)

    freq = np.fft.fftfreq(
        n,
        d=dx
    )

    ik = (
        1j
        * 2.0
        * np.pi
        * freq
    )

    k2 = -(
        2.0
        * np.pi
        * freq
    ) ** 2

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

# ============================================================
# RK4
# ============================================================

def rk4(u):

    k1 = rhs_panel(u)

    k2 = rhs_panel(
        u + 0.5 * DT * k1
    )

    k3 = rhs_panel(
        u + 0.5 * DT * k2
    )

    k4 = rhs_panel(
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
        )
        / 6.0
    )

# ============================================================
# Reference operator
# ============================================================

freq_ref = np.fft.fftfreq(
    N,
    d=dx
)

ik_ref = (
    1j
    * 2.0
    * np.pi
    * freq_ref
)

k2_ref = -(
    2.0
    * np.pi
    * freq_ref
) ** 2

def rhs_ref(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik_ref * uhat
    ).real

    uxx = np.fft.ifft(
        k2_ref * uhat
    ).real

    return (
        -(u * ux)
        + NU * uxx
    )

def rk4_ref(u):

    k1 = rhs_ref(u)

    k2 = rhs_ref(
        u + 0.5 * DT * k1
    )

    k3 = rhs_ref(
        u + 0.5 * DT * k2
    )

    k4 = rhs_ref(
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
        )
        / 6.0
    )

# ============================================================
# POD Basis Training
# ============================================================

print()
print("Building flux POD basis...")

MID = N // 2

snapshots = []

for sample in range(100):

    u = np.zeros(N)

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
                2*np.pi*k*x + phase
            )
        )

    for step in range(200):

        u = rk4_ref(u)

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
        0.999
    )
    + 1
)

print(
    "Flux POD Rank =",
    full_rank
)

# ============================================================
# Test signal
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
    u_ref = rk4_ref(u_ref)

# ============================================================
# Scaling Case
# ============================================================

def run_case(num_panels, rank=None):

    panel_size = N // num_panels

    if panel_size < 2 * W:
        return np.nan, np.nan

    panels = []

    for p in range(num_panels):

        panels.append(
            u0[
                p*panel_size :
                (p+1)*panel_size
            ].copy()
        )

    communicated = 0

    basis = None

    if rank is not None:

        rank = min(
            rank,
            U.shape[1]
        )

        basis = U[:, :rank]

    for step in range(NSTEPS):

        for p in range(num_panels - 1):

            left = panels[p]
            right = panels[p+1]

            fL = flux(
                left[-W:]
            )

            fR = flux(
                right[:W]
            )

            interface = np.concatenate(
                [fL, fR]
            )

            if rank is None:

                communicated += len(interface)

                state = interface

            else:

                centered = (
                    interface.reshape(-1,1)
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
                DT *
                beta *
                (right_flux - fL)
            )

            right[:W] += (
                DT *
                beta *
                (left_flux - fR)
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
            np.linalg.norm(u_ref)
            + 1e-14
        )
    )

    bytes_sent = (
        communicated
        * DOUBLE_BYTES
    )

    return err, bytes_sent

# ============================================================
# Run Study
# ============================================================

results = []

for npanels in PANEL_COUNTS:

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
            bytes16,
        )
    )

# ============================================================
# PDF
# ============================================================

pdf_name = "test_sfx_v29_1_scaling.pdf"

with PdfPages(pdf_name) as pdf:

    # --------------------------------------------------------
    # Error scaling plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(
        PANEL_COUNTS,
        [r[1] for r in results],
        "o-",
        label="FULL",
    )

    ax.semilogy(
        PANEL_COUNTS,
        [r[2] for r in results],
        "o-",
        label="POD22",
    )

    ax.semilogy(
        PANEL_COUNTS,
        [r[3] for r in results],
        "o-",
        label="POD16",
    )

    ax.set_xlabel("Panels")
    ax.set_ylabel("Relative Error")
    ax.set_title("SAT-POD Schwarz Scaling")
    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # Summary page
    # --------------------------------------------------------

    fig = plt.figure(figsize=(11, 8))

    plt.axis("off")

    lines = []

    lines.append("V29.1 SCALING STUDY")
    lines.append("")
    lines.append(
        "Panels  ErrFULL     Err22       Err16       BytesFULL   Bytes22     Bytes16"
    )
    lines.append("")

    for r in results:

        panels = r[0]
        err_full = r[1]
        err22 = r[2]
        err16 = r[3]
        bytes_full = int(r[4])
        bytes22 = int(r[5])
        bytes16 = int(r[6])

        lines.append(
            f"{panels:6d} "
            f"{err_full:10.3e} "
            f"{err22:10.3e} "
            f"{err16:10.3e} "
            f"{bytes_full:10d} "
            f"{bytes22:10d} "
            f"{bytes16:10d}"
        )

    txt = "\n".join(lines)

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
