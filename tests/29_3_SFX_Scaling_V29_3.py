# ============================================================
# V29.3
#
# SAT-POD Schwarz Scaling Study
#
# FULL
# POD22
# POD16
#
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================
# Parameters
# ============================================================

N = 256
L = 1.0

DT = 1e-4
NSTEPS = 1000

NU = 1e-2

W = 8

DOUBLE_BYTES = 8

PANEL_COUNTS = [2, 4, 8, 16]

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

# ============================================================
# Flux
# ============================================================

def flux(u):
    return 0.5 * u**2

# ============================================================
# Full-domain operators
# ============================================================

freq_ref = np.fft.fftfreq(
    N,
    d=dx
)

ik_ref = 1j * 2.0 * np.pi * freq_ref

k2_ref = -(2.0 * np.pi * freq_ref) ** 2

def rhs_ref(u):

    uhat = np.fft.fft(u)

    ux = np.fft.ifft(
        ik_ref * uhat
    ).real

    uxx = np.fft.ifft(
        k2_ref * uhat
    ).real

    return -(u * ux) + NU * uxx

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
        + DT *
        (k1 + 2*k2 + 2*k3 + k4)
        / 6.0
    )

# ============================================================
# Panel-local Burgers
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
    )**2

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

def rk4_panel(u):

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
        + DT *
        (
            k1
            + 2*k2
            + 2*k3
            + k4
        ) / 6.0
    )

# ============================================================
# Random training signals
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
                2*np.pi*k*x + phase
            )
        )

    return u

# ============================================================
# Build POD basis
# ============================================================

print()
print("Building flux POD basis...")

MID = N // 2

snapshots = []

for sample in range(100):

    u = random_wave()

    for step in range(200):

        u = rk4_ref(u)

        f = flux(u)

        state = np.concatenate(
            [
                f[MID-W:MID],
                f[MID:MID+W]
            ]
        )

        snapshots.append(state)

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

pod_rank = (
    np.searchsorted(
        energy,
        0.999
    )
    + 1
)

print("POD Rank =", pod_rank)

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
# Reference solution
# ============================================================

u_ref = u0.copy()

for step in range(NSTEPS):

    u_ref = rk4_ref(u_ref)

# ============================================================
# Scaling run
# ============================================================

def run_case(num_panels, rank=None):

    panel_size = N // num_panels

    if panel_size <= 2 * W:

        print(
            f"Skipping panels={num_panels}"
        )

        return (
            np.nan,
            np.nan
        )

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

                state = interface

                communicated += len(interface)

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
                (
                    right_flux
                    - fL
                )
            )

            right[:W] += (
                DT *
                beta *
                (
                    left_flux
                    - fR
                )
            )

        panels = [
            rk4_panel(panel)
            for panel in panels
        ]

    sol = np.concatenate(
        panels
    )

    error = (
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

    return error, bytes_sent

# ============================================================
# Run study
# ============================================================

results = []

for npanels in PANEL_COUNTS:

    print(f"Panels={npanels}")

    full_err, full_bytes = run_case(
        npanels,
        None
    )

    pod22_err, pod22_bytes = run_case(
        npanels,
        22
    )

    pod16_err, pod16_bytes = run_case(
        npanels,
        16
    )

    results.append(
        (
            npanels,
            full_err,
            pod22_err,
            pod16_err,
            full_bytes,
            pod22_bytes,
            pod16_bytes
        )
    )

# ============================================================
# PDF
# ============================================================

pdf_name = "test_sfx_v29_3_scaling.pdf"

with PdfPages(pdf_name) as pdf:

    # --------------------------------------------------------
    # Error plot
    # --------------------------------------------------------

    valid = [
        r for r in results
        if np.isfinite(r[1])
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(valid) > 0:

        ax.semilogy(
            [r[0] for r in valid],
            [r[1] for r in valid],
            "o-",
            label="FULL"
        )

        ax.semilogy(
            [r[0] for r in valid],
            [r[2] for r in valid],
            "o-",
            label="POD22"
        )

        ax.semilogy(
            [r[0] for r in valid],
            [r[3] for r in valid],
            "o-",
            label="POD16"
        )

    ax.set_xlabel("Panels")
    ax.set_ylabel("Relative Error")
    ax.set_title("V29.3 Scaling")
    ax.grid(True)
    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # Summary page
    # --------------------------------------------------------

    fig = plt.figure(figsize=(11, 8))

    plt.axis("off")

    lines = []

    lines.append("V29.3 SCALING STUDY")
    lines.append("")
    lines.append(
        "Panels  ErrFULL      Err22        Err16        BytesFULL   Bytes22     Bytes16"
    )
    lines.append(
        "------  -----------  -----------  -----------  ----------  ----------  ----------"
    )

    for row in results:

        panels = row[0]

        if not np.isfinite(row[1]):

            lines.append(
                f"{panels:6d}  SKIPPED"
            )

            continue

        lines.append(
            "{:6d} {:12.4e} {:12.4e} {:12.4e} {:10d} {:10d} {:10d}".format(
                panels,
                row[1],
                row[2],
                row[3],
                int(row[4]),
                int(row[5]),
                int(row[6]),
            )
        )

    plt.text(
        0.01,
        0.99,
        "\n".join(lines),
        va="top",
        family="monospace",
    )

    pdf.savefig(fig)
    plt.close(fig)

print()
print(f"Saved: {pdf_name}")
