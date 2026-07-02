import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ======================================================
# V11
# Generic 2x4 POD Schwarz
# ======================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 250
N_SCHWARZ = 5

CFL = 0.35
VX = 1.0
VY = 0.5

SIGMA = 0.05

TOLS = [1e-2,1e-3,1e-4]

NUM_INTERFACES = (
    (NROW-1)*NCOL +
    NROW*(NCOL-1)
)

OMEGA = 0.01 / np.sqrt(NUM_INTERFACES)

PDF = "09_SFX_8Panel_POD_Schwarz_V11.pdf"

# ======================================================
# Grid
# ======================================================

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

u0 = np.exp(
    -((X-0.25)**2+(Y-0.50)**2)
    /(SIGMA**2)
)

# ======================================================
# Transport
# ======================================================

def full_step(q):

    qx = q - np.roll(q,1,axis=0)
    qy = q - np.roll(q,1,axis=1)

    return q - CFL*VX*qx - CFL*VY*qy

# ======================================================
# Flux
# ======================================================

def flux(u):
    return 0.5*u*u

# ======================================================
# Split / Merge
# ======================================================

def split_panels(q):

    panels=[]

    for i in range(NROW):

        row=[]

        for j in range(NCOL):

            row.append(
                q[
                    i*PX:(i+1)*PX,
                    j*PY:(j+1)*PY
                ].copy()
            )

        panels.append(row)

    return panels

def merge_panels(P):

    rows=[]

    for i in range(NROW):

        rows.append(
            np.hstack(P[i])
        )

    return np.vstack(rows)

# ======================================================
# POD Training
# ======================================================

print("Building POD basis...")

snaps=[]

q=u0.copy()

for _ in range(500):

    q=full_step(q)

    for i in range(NROW):

        for j in range(NCOL):

            block=q[
                i*PX:(i+1)*PX,
                j*PY:(j+1)*PY
            ]

            snaps.append(block[-1,:].copy())
            snaps.append(block[:, -1].copy())

S=np.asarray(snaps)

mean_trace=S.mean(axis=0)

A=(S-mean_trace).T

U,Sigma,VT=np.linalg.svd(
    A,
    full_matrices=False
)

energy=np.cumsum(Sigma**2)
energy/=energy[-1]

def pod_rank(tol):

    target=1.0-tol**2

    return max(
        1,
        np.searchsorted(
            energy,
            target
        )+1
    )

def pod_compress(trace,tol):

    r=pod_rank(tol)

    centered=trace-mean_trace

    rec=(
        U[:,:r]
        @
        (U[:,:r].T @ centered)
        +
        mean_trace
    )

    return rec,r

# ======================================================
# Schwarz Sweep
# ======================================================

def schwarz_step(P,tol):

    rank_hist=[]
    resid_hist=[]

    for _ in range(N_SCHWARZ):

        #
        # vertical interfaces
        #

        for i in range(NROW-1):

            for j in range(NCOL):

                top=P[i][j]
                bot=P[i+1][j]

                r=(
                    flux(top[-1,:])
                    -
                    flux(bot[0,:])
                )

                rec,rank=pod_compress(
                    r,
                    tol
                )

                top[-1,:] -= OMEGA*rec
                bot[0,:] += OMEGA*rec

                rank_hist.append(rank)
                resid_hist.append(
                    np.linalg.norm(r)
                )

        #
        # horizontal interfaces
        #

        for i in range(NROW):

            for j in range(NCOL-1):

                left=P[i][j]
                right=P[i][j+1]

                r=(
                    flux(left[:,-1])
                    -
                    flux(right[:,0])
                )

                rec,rank=pod_compress(
                    r,
                    tol
                )

                left[:,-1] -= OMEGA*rec
                right[:,0] += OMEGA*rec

                rank_hist.append(rank)
                resid_hist.append(
                    np.linalg.norm(r)
                )

    q=merge_panels(P)

    q=full_step(q)

    return (
        split_panels(q),
        np.mean(rank_hist),
        resid_hist[-1]
    )

# ======================================================
# Reference
# ======================================================

ref=u0.copy()

refhist=[]

for _ in range(NT):

    ref=full_step(ref)

    refhist.append(ref.copy())

# ======================================================
# Experiments
# ======================================================

results=[]

with PdfPages(PDF) as pdf:

    for tol in TOLS:

        print("tol =",tol)

        P=split_panels(u0.copy())

        errs=[]
        ranks=[]
        residuals=[]

        for n in range(NT):

            P,rank,res=schwarz_step(
                P,
                tol
            )

            q=merge_panels(P)

            err=(
                np.linalg.norm(
                    q-refhist[n]
                )
                /
                (
                    np.linalg.norm(
                        refhist[n]
                    )
                    +1e-16
                )
            )

            errs.append(err)
            ranks.append(rank)
            residuals.append(res)

        results.append(
            (
                tol,
                errs[-1],
                np.mean(ranks),
                residuals[-1]
            )
        )

        fig,ax=plt.subplots(
            1,3,
            figsize=(12,4)
        )

        ax[0].semilogy(
            np.maximum(errs,1e-16)
        )
        ax[0].set_title("L2 Error")

        ax[1].plot(ranks)
        ax[1].set_title("POD Rank")

        ax[2].semilogy(
            np.maximum(
                residuals,
                1e-16
            )
        )
        ax[2].set_title(
            "Interface Residual"
        )

        fig.suptitle(
            f"8-Panel POD Schwarz tol={tol:.0e}"
        )

        pdf.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis("off")

    lines=[
        "SFX V11 8-PANEL POD SCHWARZ",
        "",
        "Tol FinalErr AvgRank FinalResidual",
        ""
    ]

    for tol,err,rank,resid in results:

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

print("Saved",PDF)

for r in results:
    print(r)
