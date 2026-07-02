import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# V10 : 4-Panel POD Schwarz
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

PDF = "08_SFX_4Panel_POD_Schwarz_V10.pdf"

# ==========================================================
# Grid
# ==========================================================

x = np.linspace(0, 1, NX, endpoint=False)
y = np.linspace(0, 1, NY, endpoint=False)

X, Y = np.meshgrid(x, y, indexing="ij")

u0 = np.exp(
    -(
        (X - 0.25)**2 +
        (Y - 0.50)**2
    ) / 0.05**2
)

# ==========================================================
# Transport
# ==========================================================

def full_step(q):

    qx = q - np.roll(q, 1, axis=0)
    qy = q - np.roll(q, 1, axis=1)

    return q - CFL*VX*qx - CFL*VY*qy

# ==========================================================
# 4-panel split/merge
# ==========================================================

def split4(q):

    p1 = q[:MX,:MY].copy()
    p2 = q[MX:,:MY].copy()

    p3 = q[:MX,MY:].copy()
    p4 = q[MX:,MY:].copy()

    return [p1,p2,p3,p4]

def merge4(P):

    top = np.vstack([P[0],P[1]])
    bottom = np.vstack([P[2],P[3]])

    return np.hstack([top,bottom])

# ==========================================================
# POD Training
# ==========================================================

print("Building POD basis...")

snaps=[]

q=u0.copy()

for _ in range(500):

    q=full_step(q)

    snaps.append(q[MX-1,:].copy())
    snaps.append(q[:,MY-1].copy())

S=np.asarray(snaps)

mean_trace=np.mean(S,axis=0)

A=(S-mean_trace).T

U,Sigma,VT=np.linalg.svd(
    A,
    full_matrices=False
)

energy=np.cumsum(Sigma**2)
energy/=energy[-1]

def pod_rank(tol):

    tgt=1.0-tol**2

    return max(
        1,
        np.searchsorted(
            energy,
            tgt
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

# ==========================================================
# Flux
# ==========================================================

def flux(u):

    return 0.5*u*u

# ==========================================================
# Schwarz iteration
# ==========================================================

def apply_interface(a,b,tol):

    ta=a[-1,:].copy()
    tb=b[0,:].copy()

    fa=flux(ta)
    fb=flux(tb)

    residual=fa-fb

    rec,rank=pod_compress(
        residual,
        tol
    )

    a[-1,:] -= OMEGA*rec
    b[0,:] += OMEGA*rec

    return rank,np.linalg.norm(rec)

def schwarz_step(P,tol):

    ranks=[]
    residuals=[]

    for _ in range(N_SCHWARZ):

        r,res=apply_interface(
            P[0],
            P[1],
            tol
        )
        ranks.append(r)
        residuals.append(res)

        r,res=apply_interface(
            P[2],
            P[3],
            tol
        )
        ranks.append(r)
        residuals.append(res)

        # horizontal

        top=P[0][:,-1]
        bot=P[2][:,0]

        fres=flux(top)-flux(bot)

        rec,r=pod_compress(
            fres,
            tol
        )

        P[0][:,-1]-=OMEGA*rec
        P[2][:,0]+=OMEGA*rec

        ranks.append(r)

        top=P[1][:,-1]
        bot=P[3][:,0]

        fres=flux(top)-flux(bot)

        rec,r=pod_compress(
            fres,
            tol
        )

        P[1][:,-1]-=OMEGA*rec
        P[3][:,0]+=OMEGA*rec

        ranks.append(r)

    q=merge4(P)

    q=full_step(q)

    return split4(q),np.mean(ranks),residuals[-1]

# ==========================================================
# Reference
# ==========================================================

ref=u0.copy()

refhist=[]

for _ in range(NT):

    ref=full_step(ref)

    refhist.append(ref.copy())

# ==========================================================
# Experiments
# ==========================================================

results=[]

with PdfPages(PDF) as pdf:

    for tol in TOLS:

        P=split4(u0.copy())

        errs=[]
        ranks=[]
        resids=[]

        for n in range(NT):

            P,rank,res=schwarz_step(
                P,
                tol
            )

            q=merge4(P)

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
            resids.append(res)

        results.append(
            (
                tol,
                errs[-1],
                np.mean(ranks),
                resids[-1]
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
            np.maximum(resids,1e-16)
        )
        ax[2].set_title("Residual")

        fig.suptitle(
            f"4 Panel POD Schwarz tol={tol:.0e}"
        )

        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(figsize=(11,8))

    plt.axis("off")

    lines = [
        "SFX V10 4-PANEL POD SCHWARZ",
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

for r in results:
    print(r)
