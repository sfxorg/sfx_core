import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# SFX V10.1
# 4-Panel POD Flux-Residual Schwarz
# ==========================================================

NX = 128
NY = 128

MX = NX // 2
MY = NY // 2

NT = 250
N_SCHWARZ = 5

CFL = 0.35
VX = 1.0
VY = 0.5

OMEGA = 0.01

TOLS = [1e-2, 1e-3, 1e-4]

PDF = "08_SFX_4Panel_POD_Schwarz_V10_1.pdf"

# ==========================================================
# Grid
# ==========================================================

x = np.linspace(0, 1, NX, endpoint=False)
y = np.linspace(0, 1, NY, endpoint=False)

X, Y = np.meshgrid(x, y, indexing="ij")

u0 = np.exp(
    -(
        (X - 0.25) ** 2
        + (Y - 0.50) ** 2
    ) / 0.05**2
)

# ==========================================================
# Transport
# ==========================================================

def full_step(q):

    qx = q - np.roll(q, 1, axis=0)
    qy = q - np.roll(q, 1, axis=1)

    return q - CFL * VX * qx - CFL * VY * qy

# ==========================================================
# Split / Merge
# ==========================================================

def split4(q):

    p1 = q[:MX, :MY].copy()
    p2 = q[MX:, :MY].copy()

    p3 = q[:MX, MY:].copy()
    p4 = q[MX:, MY:].copy()

    return [p1, p2, p3, p4]


def merge4(P):

    upper = np.hstack([P[0], P[2]])
    lower = np.hstack([P[1], P[3]])

    return np.vstack([upper, lower])

# ==========================================================
# Flux
# ==========================================================

def flux(u):
    return 0.5 * u * u

# ==========================================================
# POD Training (64-point interfaces)
# ==========================================================

print("Building POD basis...")

snaps = []

q = u0.copy()

for _ in range(500):

    q = full_step(q)

    # vertical interfaces

    snaps.append(q[MX - 1, :MY].copy())
    snaps.append(q[MX - 1, MY:].copy())

    # horizontal interfaces

    snaps.append(q[:MX, MY - 1].copy())
    snaps.append(q[MX:, MY - 1].copy())

S = np.asarray(snaps)

mean_trace = np.mean(S, axis=0)

A = (S - mean_trace).T

U, Sigma, VT = np.linalg.svd(
    A,
    full_matrices=False
)

energy = np.cumsum(Sigma**2)
energy /= energy[-1]


def pod_rank(tol):

    target = 1.0 - tol**2

    return max(
        1,
        np.searchsorted(
            energy,
            target
        ) + 1
    )


def pod_compress(trace, tol):

    r = pod_rank(tol)

    centered = trace - mean_trace

    rec = (
        U[:, :r]
        @
        (
            U[:, :r].T @ centered
        )
        +
        mean_trace
    )

    return rec, r

# ==========================================================
# Interface Operators
# ==========================================================

def vertical_interface(left_panel,
                       right_panel,
                       tol):

    uL = left_panel[-1, :]
    uR = right_panel[0, :]

    residual = flux(uL) - flux(uR)

    rec, rank = pod_compress(
        residual,
        tol
    )

    left_panel[-1, :] -= OMEGA * rec
    right_panel[0, :] += OMEGA * rec

    return rank, np.linalg.norm(rec)


def horizontal_interface(top_panel,
                         bottom_panel,
                         tol):

    uT = top_panel[:, -1]
    uB = bottom_panel[:, 0]

    residual = flux(uT) - flux(uB)

    rec, rank = pod_compress(
        residual,
        tol
    )

    top_panel[:, -1] -= OMEGA * rec
    bottom_panel[:, 0] += OMEGA * rec

    return rank, np.linalg.norm(rec)

# ==========================================================
# Schwarz Iteration
# ==========================================================

def schwarz_step(P, tol):

    rank_hist = []
    residual_hist = []

    for _ in range(N_SCHWARZ):

        # P1 | P2

        r, res = vertical_interface(
            P[0],
            P[1],
            tol
        )

        rank_hist.append(r)
        residual_hist.append(res)

        # P3 | P4

        r, res = vertical_interface(
            P[2],
            P[3],
            tol
        )

        rank_hist.append(r)
        residual_hist.append(res)

        # P1 over P3

        r, res = horizontal_interface(
            P[0],
            P[2],
            tol
        )

        rank_hist.append(r)
        residual_hist.append(res)

        # P2 over P4

        r, res = horizontal_interface(
            P[1],
            P[3],
            tol
        )

        rank_hist.append(r)
        residual_hist.append(res)

    q = merge4(P)

    q = full_step(q)

    return (
        split4(q),
        np.mean(rank_hist),
        residual_hist[-1]
    )

# ==========================================================
# Reference
# ==========================================================

ref = u0.copy()

ref_hist = []

for _ in range(NT):

    ref = full_step(ref)

    ref_hist.append(ref.copy())

# ==========================================================
# Experiments
# ==========================================================

results = []

with PdfPages(PDF) as pdf:

    for tol in TOLS:

        print("tol =", tol)

        P = split4(u0.copy())

        errs = []
        ranks = []
        residuals = []

        for n in range(NT):

            P, rank, resid = schwarz_step(
                P,
                tol
            )

            q = merge4(P)

            err = (
                np.linalg.norm(
                    q - ref_hist[n]
                )
                /
                (
                    np.linalg.norm(
                        ref_hist[n]
                    )
                    + 1e-16
                )
            )

            errs.append(err)
            ranks.append(rank)
            residuals.append(resid)

        results.append(
            (
                tol,
                errs[-1],
                np.mean(ranks),
                residuals[-1]
            )
        )

        fig, ax = plt.subplots(
            1,
            3,
            figsize=(12, 4)
        )

        ax[0].semilogy(
            np.maximum(errs, 1e-16)
        )
        ax[0].set_title("L2 Error")

        ax[1].plot(ranks)
        ax[1].set_title("POD Rank")

        ax[2].semilogy(
            np.maximum(residuals, 1e-16)
        )
        ax[2].set_title("Residual")

        fig.suptitle(
            f"4-Panel POD Schwarz tol={tol:.0e}"
        )

        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(
        figsize=(11, 8)
    )

    plt.axis("off")

    lines = [
        "SFX V10.1 4-PANEL POD SCHWARZ",
        "",
        "Tol FinalErr AvgRank FinalResidual",
        ""
    ]

    for tol, err, rank, resid in results:

        lines.append(
            f"{tol:.0e} "
            f"{err:.6e} "
            f"{rank:.2f} "
            f"{resid:.6e}"
        )

    plt.text(
        0.02,
        0.98,
        "\n".join(lines),
        family="monospace",
        va="top"
    )

    pdf.savefig(fig)
    plt.close(fig)

print("Saved", PDF)

for row in results:
    print(row)
