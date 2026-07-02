import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# V15
# SAT Flux Schwarz
# ==========================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 250

N_SCHWARZ = 10

CFL = 0.35
VX = 1.0
VY = 0.5

SIGMA = 0.05

OMEGA = 0.01
TOL = 1e-3

PDF = "13_SFX_SAT_Flux_Schwarz_V15.pdf"

# ==========================================================
# GRID
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
# TRANSPORT
# ==========================================================

def full_step(q):

    qx = q - np.roll(q,1,axis=0)
    qy = q - np.roll(q,1,axis=1)

    return q - CFL*VX*qx - CFL*VY*qy

# ==========================================================
# FLUX
# ==========================================================

def flux(u):
    return 0.5*u*u

def sat_flux(uL,uR):

    fL = flux(uL)
    fR = flux(uR)

    alpha = np.maximum(
        np.abs(uL),
        np.abs(uR)
    )

    return (
        0.5*(fL + fR)
        -
        0.5*alpha*(uR-uL)
    )

# ==========================================================
# SPLIT MERGE
# ==========================================================

def split_panels(q):

    P=[]

    for i in range(NROW):

        row=[]

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

    rows=[]

    for i in range(NROW):
        rows.append(np.hstack(P[i]))

    return np.vstack(rows)

# ==========================================================
# POD TRAINING
# ==========================================================

print("Building POD bases...")

snaps_h=[]
snaps_v=[]

q=u0.copy()

for _ in range(500):

    q=full_step(q)

    for i in range(NROW):

        for j in range(NCOL):

            block=q[
                i*PX:(i+1)*PX,
                j*PY:(j+1)*PY
            ]

            snaps_h.append(block[-1,:].copy())
            snaps_v.append(block[:,-1].copy())

Sh=np.asarray(snaps_h)
Sv=np.asarray(snaps_v)

mean_h=Sh.mean(axis=0)
mean_v=Sv.mean(axis=0)

Ah=(Sh-mean_h).T
Av=(Sv-mean_v).T

Uh,SigH,_=np.linalg.svd(
    Ah,
    full_matrices=False
)

Uv,SigV,_=np.linalg.svd(
    Av,
    full_matrices=False
)

EH=np.cumsum(SigH**2)
EH/=EH[-1]

EV=np.cumsum(SigV**2)
EV/=EV[-1]

def pod_rank(E):

    target=1.0-TOL**2

    return max(
        1,
        np.searchsorted(E,target)+1
    )

def pod_h(v):

    r=pod_rank(EH)

    rec=(
        Uh[:,:r]
        @
        (
            Uh[:,:r].T
            @
            (v-mean_h)
        )
        +
        mean_h
    )

    return rec,r

def pod_v(v):

    r=pod_rank(EV)

    rec=(
        Uv[:,:r]
        @
        (
            Uv[:,:r].T
            @
            (v-mean_v)
        )
        +
        mean_v
    )

    return rec,r

# ==========================================================
# SAT SCHWARZ
# ==========================================================

def schwarz_step(P):

    residuals=[]
    hranks=[]
    vranks=[]

    for _ in range(N_SCHWARZ):

        # vertical interfaces

        for i in range(NROW-1):

            for j in range(NCOL):

                A=P[i][j]
                B=P[i+1][j]

                uL=A[-1,:]
                uR=B[0,:]

                fs=sat_flux(uL,uR)

                resid=flux(uL)-fs

                rec,r=pod_h(resid)

                A[-1,:]-=OMEGA*rec
                B[0,:]+=OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

                hranks.append(r)

        # horizontal interfaces

        for i in range(NROW):

            for j in range(NCOL-1):

                A=P[i][j]
                B=P[i][j+1]

                uL=A[:,-1]
                uR=B[:,0]

                fs=sat_flux(uL,uR)

                resid=flux(uL)-fs

                rec,r=pod_v(resid)

                A[:,-1]-=OMEGA*rec
                B[:,0]+=OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

                vranks.append(r)

    q=merge_panels(P)

    q=full_step(q)

    return (
        split_panels(q),
        np.mean(hranks),
        np.mean(vranks),
        residuals[-1]
    )

# ==========================================================
# REFERENCE
# ==========================================================

ref=u0.copy()

refhist=[]

for _ in range(NT):

    ref=full_step(ref)

    refhist.append(ref.copy())

# ==========================================================
# RUN
# ==========================================================

results=[]

with PdfPages(PDF) as pdf:

    P=split_panels(u0.copy())

    errs=[]
    hr=[]
    vr=[]
    res=[]

    for n in range(NT):

        (
            P,
            hrank,
            vrank,
            resid
        ) = schwarz_step(P)

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
        hr.append(hrank)
        vr.append(vrank)
        res.append(resid)

    results.append(
        (
            errs[-1],
            np.mean(hr),
            np.mean(vr),
            res[-1]
        )
    )

    fig,ax=plt.subplots(
        1,4,
        figsize=(15,4)
    )

    ax[0].semilogy(np.maximum(errs,1e-16))
    ax[0].set_title("L2 Error")

    ax[1].plot(hr)
    ax[1].set_title("H Rank")

    ax[2].plot(vr)
    ax[2].set_title("V Rank")

    ax[3].semilogy(
        np.maximum(res,1e-16)
    )
    ax[3].set_title("SAT Residual")

    pdf.savefig(fig)
    plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis("off")

    lines=[
        "SFX V15 SAT FLUX SCHWARZ",
        "",
        "FinalErr HRank VRank FinalResidual",
        ""
    ]

    for e,h,v,r in results:

        lines.append(
            f"{e:.6e} "
            f"{h:.2f} "
            f"{v:.2f} "
            f"{r:.6e}"
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
