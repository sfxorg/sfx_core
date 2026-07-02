import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# V14
# State Schwarz vs Flux Schwarz
# ==========================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 250

MAX_SCHWARZ = 20

CFL = 0.35
VX = 1.0
VY = 0.5

SIGMA = 0.05

OMEGA = 0.003
TOL = 1e-3

PDF = "12_SFX_State_vs_Flux_Schwarz_V14.pdf"

# ==========================================================
# GRID
# ==========================================================

x=np.linspace(0,1,NX,endpoint=False)
y=np.linspace(0,1,NY,endpoint=False)

X,Y=np.meshgrid(x,y,indexing="ij")

u0=np.exp(
    -((X-0.25)**2+(Y-0.50)**2)
    /(SIGMA**2)
)

# ==========================================================
# TRANSPORT
# ==========================================================

def full_step(q):

    qx=q-np.roll(q,1,axis=0)
    qy=q-np.roll(q,1,axis=1)

    return q-CFL*VX*qx-CFL*VY*qy

# ==========================================================
# FLUX
# ==========================================================

def flux(u):
    return 0.5*u*u

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
# POD BASIS
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

Uh,Sigma_h,_=np.linalg.svd(
    Ah,
    full_matrices=False
)

Uv,Sigma_v,_=np.linalg.svd(
    Av,
    full_matrices=False
)

energy_h=np.cumsum(Sigma_h**2)
energy_h/=energy_h[-1]

energy_v=np.cumsum(Sigma_v**2)
energy_v/=energy_v[-1]

def pod_rank(energy):

    target=1.0-TOL**2

    return max(
        1,
        np.searchsorted(
            energy,
            target
        )+1
    )

def pod_h(v):

    r=pod_rank(energy_h)

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

    r=pod_rank(energy_v)

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
# SCHWARZ
# ==========================================================

def schwarz_step(P, mode):

    residuals=[]

    for _ in range(MAX_SCHWARZ):

        # vertical interfaces

        for i in range(NROW-1):

            for j in range(NCOL):

                A=P[i][j]
                B=P[i+1][j]

                if mode=="STATE":

                    resid=(
                        A[-1,:]
                        -
                        B[0,:]
                    )

                else:

                    resid=(
                        flux(A[-1,:])
                        -
                        flux(B[0,:])
                    )

                rec,_=pod_h(resid)

                A[-1,:]-=OMEGA*rec
                B[0,:]+=OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

        # horizontal interfaces

        for i in range(NROW):

            for j in range(NCOL-1):

                A=P[i][j]
                B=P[i][j+1]

                if mode=="STATE":

                    resid=(
                        A[:,-1]
                        -
                        B[:,0]
                    )

                else:

                    resid=(
                        flux(A[:,-1])
                        -
                        flux(B[:,0])
                    )

                rec,_=pod_v(resid)

                A[:,-1]-=OMEGA*rec
                B[:,0]+=OMEGA*rec

                residuals.append(
                    np.linalg.norm(resid)
                )

    q=merge_panels(P)
    q=full_step(q)

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

    ref=full_step(ref)

    refhist.append(ref.copy())

# ==========================================================
# EXPERIMENTS
# ==========================================================

results=[]

with PdfPages(PDF) as pdf:

    for mode in ["STATE","FLUX"]:

        print(mode)

        P=split_panels(u0.copy())

        errs=[]
        residuals=[]

        for n in range(NT):

            P,res=schwarz_step(
                P,
                mode
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
            residuals.append(res)

        results.append(
            (
                mode,
                errs[-1],
                residuals[-1]
            )
        )

        fig,ax=plt.subplots(
            1,2,
            figsize=(10,4)
        )

        ax[0].semilogy(
            np.maximum(errs,1e-16)
        )
        ax[0].set_title(
            "L2 Error"
        )

        ax[1].semilogy(
            np.maximum(
                residuals,
                1e-16
            )
        )
        ax[1].set_title(
            "Residual"
        )

        fig.suptitle(mode)

        pdf.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(11,8))

    plt.axis("off")

    lines=[
        "SFX V14 STATE VS FLUX",
        "",
        "Mode FinalErr FinalResidual",
        ""
    ]

    for mode,err,res in results:

        lines.append(
            f"{mode:5s} "
            f"{err:.6e} "
            f"{res:.6e}"
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
