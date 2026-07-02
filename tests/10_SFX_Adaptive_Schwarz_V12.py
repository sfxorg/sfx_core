# 10_SFX_Adaptive_Schwarz_V12.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# Configuration
# ==========================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 250

MAX_SCHWARZ = 50
RESIDUAL_TARGET = 1.0e-5

CFL = 0.35
VX = 1.0
VY = 0.5

SIGMA = 0.05

TOLS = [1e-2, 1e-3, 1e-4]

NUM_INTERFACES = (
    (NROW - 1) * NCOL +
    NROW * (NCOL - 1)
)

OMEGA = 0.01 / np.sqrt(NUM_INTERFACES)

PDF = "10_SFX_Adaptive_Schwarz_V12.pdf"

# ==========================================================
# Grid
# ==========================================================

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

u0 = np.exp(
    -((X-0.25)**2 +
      (Y-0.50)**2)
    / SIGMA**2
)

# ==========================================================
# Transport
# ==========================================================

def full_step(q):

    qx = q - np.roll(q,1,axis=0)
    qy = q - np.roll(q,1,axis=1)

    return q - CFL*VX*qx - CFL*VY*qy

# ==========================================================
# Flux
# ==========================================================

def flux(u):
    return 0.5*u*u

# ==========================================================
# Split / Merge
# ==========================================================

def split_panels(q):

    P = []

    for i in range(NROW):

        row = []

        for j in range(NCOL):

            row.append(
                q[
                    i*PX:(i+1)*PX,
                    j*PY:(j+1)*PY
                ].copy()
            )

        P.append(row)

    return P


def merge_panels(P):

    rows = []

    for i in range(NROW):
        rows.append(np.hstack(P[i]))

    return np.vstack(rows)

# ==========================================================
# POD training
# ==========================================================

print("Building POD bases...")

snaps_h = []
snaps_v = []

q = u0.copy()

for _ in range(500):

    q = full_step(q)

    for i in range(NROW):

        for j in range(NCOL):

            block = q[
                i*PX:(i+1)*PX,
                j*PY:(j+1)*PY
            ]

            snaps_h.append(
                block[-1,:].copy()
            )

            snaps_v.append(
                block[:,-1].copy()
            )

# horizontal POD

Sh = np.asarray(snaps_h)

mean_h = Sh.mean(axis=0)

Ah = (Sh - mean_h).T

Uh,Sigma_h,_ = np.linalg.svd(
    Ah,
    full_matrices=False
)

energy_h = np.cumsum(Sigma_h**2)
energy_h /= energy_h[-1]

# vertical POD

Sv = np.asarray(snaps_v)

mean_v = Sv.mean(axis=0)

Av = (Sv - mean_v).T

Uv,Sigma_v,_ = np.linalg.svd(
    Av,
    full_matrices=False
)

energy_v = np.cumsum(Sigma_v**2)
energy_v /= energy_v[-1]

# ==========================================================
# POD compression
# ==========================================================

def pod_rank(energy,tol):

    target = 1.0 - tol**2

    return max(
        1,
        np.searchsorted(
            energy,
            target
        ) + 1
    )

def pod_compress_h(trace,tol):

    r = pod_rank(
        energy_h,
        tol
    )

    centered = trace - mean_h

    rec = (
        Uh[:,:r]
        @
        (
            Uh[:,:r].T @ centered
        )
        +
        mean_h
    )

    return rec,r

def pod_compress_v(trace,tol):

    r = pod_rank(
        energy_v,
        tol
    )

    centered = trace - mean_v

    rec = (
        Uv[:,:r]
        @
        (
            Uv[:,:r].T @ centered
        )
        +
        mean_v
    )

    return rec,r

# ==========================================================
# Adaptive Schwarz
# ==========================================================

def adaptive_schwarz(P,tol):

    rank_h_hist = []
    rank_v_hist = []

    current_residual = 1.0
    niters = 0

    while (
        current_residual > RESIDUAL_TARGET
        and
        niters < MAX_SCHWARZ
    ):

        residuals = []

        # vertical interfaces

        for i in range(NROW-1):

            for j in range(NCOL):

                top = P[i][j]
                bottom = P[i+1][j]

                resid = (
                    flux(top[-1,:])
                    -
                    flux(bottom[0,:])
                )

                rec,r = pod_compress_h(
                    resid,
                    tol
                )

                top[-1,:] -= OMEGA*rec
                bottom[0,:] += OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

                rank_h_hist.append(r)

        # horizontal interfaces

        for i in range(NROW):

            for j in range(NCOL-1):

                left = P[i][j]
                right = P[i][j+1]

                resid = (
                    flux(left[:,-1])
                    -
                    flux(right[:,0])
                )

                rec,r = pod_compress_v(
                    resid,
                    tol
                )

                left[:,-1] -= OMEGA*rec
                right[:,0] += OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

                rank_v_hist.append(r)

        current_residual = max(residuals)

        niters += 1

    q = merge_panels(P)

    q = full_step(q)

    return (
        split_panels(q),
        np.mean(rank_h_hist),
        np.mean(rank_v_hist),
        current_residual,
        niters
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
# Experiment
# ==========================================================

results = []

with PdfPages(PDF) as pdf:

    for tol in TOLS:

        print("tol =",tol)

        P = split_panels(u0.copy())

        errs = []
        hranks = []
        vranks = []
        residuals = []
        iters_hist = []

        for n in range(NT):

            (
                P,
                hrank,
                vrank,
                resid,
                niters
            ) = adaptive_schwarz(
                P,
                tol
            )

            q = merge_panels(P)

            err = (
                np.linalg.norm(
                    q-ref_hist[n]
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
            hranks.append(hrank)
            vranks.append(vrank)
            residuals.append(resid)
            iters_hist.append(niters)

        results.append(
            (
                tol,
                errs[-1],
                np.mean(hranks),
                np.mean(vranks),
                residuals[-1],
                np.mean(iters_hist),
                np.max(iters_hist)
            )
        )

        fig,ax = plt.subplots(
            1,5,
            figsize=(18,4)
        )

        ax[0].semilogy(
            np.maximum(errs,1e-16)
        )
        ax[0].set_title("L2 Error")

        ax[1].plot(hranks)
        ax[1].set_title("H Rank")

        ax[2].plot(vranks)
        ax[2].set_title("V Rank")

        ax[3].semilogy(
            np.maximum(
                residuals,
                1e-16
            )
        )
        ax[3].set_title("Residual")

        ax[4].plot(iters_hist)
        ax[4].set_title("Schwarz Iter")

        fig.suptitle(
            f"Adaptive Schwarz tol={tol:.0e}"
        )

        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(figsize=(11,8))

    plt.axis("off")

    lines = [
        "SFX V12 ADAPTIVE SCHWARZ",
        "",
        "Tol FinalErr HRank VRank Resid AvgIter MaxIter",
        ""
    ]

    for (
        tol,
        err,
        hrank,
        vrank,
        resid,
        avgiter,
        maxiter
    ) in results:

        lines.append(
            f"{tol:.0e} "
            f"{err:.6e} "
            f"{hrank:.2f} "
            f"{vrank:.2f} "
            f"{resid:.6e} "
            f"{avgiter:.2f} "
            f"{maxiter:d}"
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

for r in results:
    print(r)
