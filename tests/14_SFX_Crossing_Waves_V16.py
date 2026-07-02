import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# V16
# Crossing-Wave SAT Flux Schwarz Test
# ==========================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 300

N_SCHWARZ = 10

CFL = 0.35

SIGMA = 0.04

OMEGA = 0.01
TOL = 1e-3

PDF = "14_SFX_Crossing_Waves_V16.pdf"

# ==========================================================
# GRID
# ==========================================================

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(
    x,
    y,
    indexing="ij"
)

# ==========================================================
# TWO CROSSING WAVES
# ==========================================================

g1 = np.exp(
    -(
        (X-0.25)**2+
        (Y-0.50)**2
    )/(SIGMA**2)
)

g2 = np.exp(
    -(
        (X-0.75)**2+
        (Y-0.50)**2
    )/(SIGMA**2)
)

u0 = g1 + g2

# ==========================================================
# TRANSPORT
# ==========================================================

def full_step(q):

    qx = q - np.roll(q,1,axis=0)
    qy = q - np.roll(q,1,axis=1)

    return (
        q
        - CFL*1.0*qx
        - CFL*0.5*qy
    )

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
        0.5*(fL+fR)
        -
        0.5*alpha*(uR-uL)
    )

# ==========================================================
# SPLIT / MERGE
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

            snaps_h.append(
                block[-1,:].copy()
            )

            snaps_v.append(
                block[:,-1].copy()
            )

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

    return max(
        1,
        np.searchsorted(
            E,
            1.0-TOL**2
        )+1
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

    hr=[]
    vr=[]

    for _ in range(N_SCHWARZ):

        # vertical

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

                hr.append(r)

        # horizontal

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

                vr.append(r)

    q=merge_panels(P)

    q=full_step(q)

    return (
        split_panels(q),
        np.mean(hr),
        np.mean(vr),
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

P=split_panels(u0.copy())

errs=[]
res=[]
hrank=[]
vrank=[]

with PdfPages(PDF) as pdf:

    for n in range(NT):

        P,h,v,r = schwarz_step(P)

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
        res.append(r)
        hrank.append(h)
        vrank.append(v)

    fig,ax=plt.subplots(
        1,4,
        figsize=(15,4)
    )

    ax[0].semilogy(errs)
    ax[0].set_title("L2 Error")

    ax[1].plot(hrank)
    ax[1].set_title("H Rank")

    ax[2].plot(vrank)
    ax[2].set_title("V Rank")

    ax[3].semilogy(np.maximum(res,1e-16))
    ax[3].set_title("SAT Residual")

    pdf.savefig(fig)
    plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis("off")

    summary = [
        "SFX V16 CROSSING WAVES",
        "",
        f"FinalErr      = {errs[-1]:.6e}",
        f"FinalResidual = {res[-1]:.6e}",
        f"HRank         = {np.mean(hrank):.2f}",
        f"VRank         = {np.mean(vrank):.2f}",
    ]

    plt.text(
        0.02,
        0.98,
        "\n".join(summary),
        family="monospace",
        va="top"
    )

    pdf.savefig(fig)
    plt.close(fig)

print("Saved", PDF)

print("FinalErr =", errs[-1])
print("FinalResidual =", res[-1])
print("HRank =", np.mean(hrank))
print("VRank =", np.mean(vrank))
