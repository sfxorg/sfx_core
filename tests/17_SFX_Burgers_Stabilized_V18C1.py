import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# V18C.1
# Stabilized Nonlinear Burgers
# ==========================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 250

N_SCHWARZ = 3

CFL = 0.05

OMEGA = 0.001

TOL = 1e-3

SIGMA = 0.04

PDF = "17_SFX_Burgers_Stabilized_V18C1.pdf"

# ==========================================================
# GRID
# ==========================================================

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

# ==========================================================
# LOW-AMPLITUDE BROADBAND PACKETS
# ==========================================================

r1 = ((X-0.25)**2 + (Y-0.50)**2)
r2 = ((X-0.75)**2 + (Y-0.50)**2)

env1 = np.exp(-r1/(SIGMA**2))
env2 = np.exp(-r2/(SIGMA**2))

wave1 = (
      np.sin(8*np.pi*X)
    + 0.50*np.sin(16*np.pi*X)
    + 0.25*np.sin(32*np.pi*X)
)

wave2 = (
      np.sin(12*np.pi*X)
    + 0.40*np.sin(24*np.pi*X)
    + 0.20*np.sin(48*np.pi*X)
)

u0 = 0.10 * (env1*wave1 + env2*wave2)

# ==========================================================
# FLUX
# ==========================================================

def flux(u):
    return 0.5*u*u

# ==========================================================
# BURGERS STEP
# ==========================================================

def burgers_step(q):

    qR = np.roll(q,-1,axis=0)

    fL = flux(q)
    fR = flux(qR)

    alpha = np.maximum(
        np.abs(q),
        np.abs(qR)
    )

    F = (
        0.5*(fL+fR)
        -
        0.5*alpha*(qR-q)
    )

    divF = F - np.roll(F,1,axis=0)

    qnew = q - CFL*divF

    qnew = np.nan_to_num(
        qnew,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0
    )

    qnew = np.clip(
        qnew,
        -10.0,
        10.0
    )

    return qnew

# ==========================================================
# SAT INTERFACE FLUX
# ==========================================================

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
# PANEL UTILITIES
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

for _ in range(300):

    q=burgers_step(q)

    for i in range(NROW):

        for j in range(NCOL):

            blk=q[
                i*PX:(i+1)*PX,
                j*PY:(j+1)*PY
            ]

            snaps_h.append(blk[-1,:].copy())
            snaps_v.append(blk[:,-1].copy())

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

def rank_from_tol(E):

    target=1.0-TOL**2

    return max(
        1,
        np.searchsorted(E,target)+1
    )

RH = rank_from_tol(EH)
RV = rank_from_tol(EV)

def pod_h(v):

    rec=(
        Uh[:,:RH]
        @
        (
            Uh[:,:RH].T
            @
            (v-mean_h)
        )
        +
        mean_h
    )

    return rec

def pod_v(v):

    rec=(
        Uv[:,:RV]
        @
        (
            Uv[:,:RV].T
            @
            (v-mean_v)
        )
        +
        mean_v
    )

    return rec

# ==========================================================
# SCHWARZ
# ==========================================================

def schwarz_step(P):

    residuals=[]

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

                resid=np.clip(
                    resid,
                    -0.05,
                    0.05
                )

                rec=pod_h(resid)

                A[-1,:]-=OMEGA*rec
                B[0,:]+=OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

        # horizontal

        for i in range(NROW):

            for j in range(NCOL-1):

                A=P[i][j]
                B=P[i][j+1]

                uL=A[:,-1]
                uR=B[:,0]

                fs=sat_flux(uL,uR)

                resid=flux(uL)-fs

                resid=np.clip(
                    resid,
                    -0.05,
                    0.05
                )

                rec=pod_v(resid)

                A[:,-1]-=OMEGA*rec
                B[:,0]+=OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

    q=merge_panels(P)

    q=burgers_step(q)

    return (
        split_panels(q),
        residuals[-1]
    )

# ==========================================================
# REFERENCE
# ==========================================================

ref=u0.copy()

refhist=[]

for _ in range(NT):

    ref=burgers_step(ref)

    refhist.append(ref.copy())

# ==========================================================
# RUN
# ==========================================================

P=split_panels(u0.copy())

errs=[]
res=[]

with PdfPages(PDF) as pdf:

    for n in range(NT):

        P,r=schwarz_step(P)

        q=merge_panels(P)

        err=(
            np.linalg.norm(q-refhist[n])
            /
            (
                np.linalg.norm(refhist[n])
                +1e-16
            )
        )

        errs.append(err)
        res.append(r)

    fig,ax=plt.subplots(
        1,2,
        figsize=(10,4)
    )

    ax[0].semilogy(
        np.maximum(errs,1e-16)
    )
    ax[0].set_title("L2 Error")

    ax[1].semilogy(
        np.maximum(res,1e-16)
    )
    ax[1].set_title("SAT Residual")

    pdf.savefig(fig)
    plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis("off")

    summary=[
        "SFX V18C.1 STABILIZED BURGERS",
        "",
        f"FinalErr      = {errs[-1]:.6e}",
        f"FinalResidual = {res[-1]:.6e}",
        f"HRank         = {RH}",
        f"VRank         = {RV}"
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

print("Saved",PDF)
print("FinalErr =",errs[-1])
print("FinalResidual =",res[-1])
print("HRank =",RH)
print("VRank =",RV)
