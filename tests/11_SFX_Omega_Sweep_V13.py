import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =====================================================
# V13
# OMEGA SWEEP STUDY
# =====================================================

NX = 128
NY = 128

NROW = 2
NCOL = 4

PX = NX // NROW
PY = NY // NCOL

NT = 250

MAX_SCHWARZ = 50
RESIDUAL_TARGET = 1e-5

CFL = 0.35
VX = 1.0
VY = 0.5

SIGMA = 0.05

TOL = 1e-3

OMEGA_LIST = [
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1
]

PDF = "11_SFX_Omega_Sweep_V13.pdf"

# =====================================================
# GRID
# =====================================================

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

u0 = np.exp(
    -((X-0.25)**2 +
      (Y-0.5)**2)
    / SIGMA**2
)

# =====================================================
# TRANSPORT
# =====================================================

def full_step(q):

    qx = q - np.roll(q,1,axis=0)
    qy = q - np.roll(q,1,axis=1)

    return q - CFL*VX*qx - CFL*VY*qy

# =====================================================
# FLUX
# =====================================================

def flux(u):
    return 0.5*u*u

# =====================================================
# SPLIT MERGE
# =====================================================

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

# =====================================================
# POD TRAINING
# =====================================================

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

# =====================================================
# POD
# =====================================================

def pod_rank(energy,tol):

    target=1.0-tol**2

    return max(
        1,
        np.searchsorted(
            energy,
            target
        )+1
    )

def pod_h(trace):

    r=pod_rank(
        energy_h,
        TOL
    )

    centered=trace-mean_h

    rec=(
        Uh[:,:r]
        @
        (
            Uh[:,:r].T
            @
            centered
        )
        +
        mean_h
    )

    return rec,r

def pod_v(trace):

    r=pod_rank(
        energy_v,
        TOL
    )

    centered=trace-mean_v

    rec=(
        Uv[:,:r]
        @
        (
            Uv[:,:r].T
            @
            centered
        )
        +
        mean_v
    )

    return rec,r

# =====================================================
# ADAPTIVE SCHWARZ
# =====================================================

def adaptive_schwarz(P,omega):

    niters=0

    hr=[]
    vr=[]

    residual=1.0

    while (
        residual > RESIDUAL_TARGET
        and
        niters < MAX_SCHWARZ
    ):

        residuals=[]

        # vertical

        for i in range(NROW-1):

            for j in range(NCOL):

                tp=P[i][j]
                bt=P[i+1][j]

                r=(
                    flux(tp[-1,:])
                    -
                    flux(bt[0,:])
                )

                rec,rank=pod_h(r)

                tp[-1,:]-=omega*rec
                bt[0,:]+=omega*rec

                hr.append(rank)

                residuals.append(
                    np.linalg.norm(r)
                )

        # horizontal

        for i in range(NROW):

            for j in range(NCOL-1):

                lf=P[i][j]
                rt=P[i][j+1]

                r=(
                    flux(lf[:,-1])
                    -
                    flux(rt[:,0])
                )

                rec,rank=pod_v(r)

                lf[:,-1]-=omega*rec
                rt[:,0]+=omega*rec

                vr.append(rank)

                residuals.append(
                    np.linalg.norm(r)
                )

        residual=max(residuals)

        niters+=1

    q=merge_panels(P)

    q=full_step(q)

    return (
        split_panels(q),
        residual,
        niters,
        np.mean(hr),
        np.mean(vr)
    )

# =====================================================
# REFERENCE
# =====================================================

ref=u0.copy()

refhist=[]

for _ in range(NT):

    ref=full_step(ref)

    refhist.append(
        ref.copy()
    )

# =====================================================
# EXPERIMENT
# =====================================================

results=[]

with PdfPages(PDF) as pdf:

    for omega in OMEGA_LIST:

        print("omega =",omega)

        P=split_panels(
            u0.copy()
        )

        errs=[]
        residuals=[]
        iters=[]
        hranks=[]
        vranks=[]

        stable=True

        for n in range(NT):

            try:

                (
                    P,
                    resid,
                    nit,
                    hr,
                    vr
                ) = adaptive_schwarz(
                    P,
                    omega
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

                residuals.append(resid)
                iters.append(nit)

                hranks.append(hr)
                vranks.append(vr)

            except Exception:

                stable=False
                break

        final_err=np.nan if not stable else errs[-1]
        final_res=np.nan if not stable else residuals[-1]

        results.append(
            (
                omega,
                final_err,
                final_res,
                np.mean(iters) if stable else np.nan,
                np.mean(hranks) if stable else np.nan,
                np.mean(vranks) if stable else np.nan
            )
        )

        if stable:

            fig,ax=plt.subplots(
                1,3,
                figsize=(12,4)
            )

            ax[0].semilogy(
                np.maximum(
                    errs,
                    1e-16
                )
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

            ax[2].plot(iters)

            ax[2].set_title(
                "Iterations"
            )

            fig.suptitle(
                f"Omega={omega}"
            )

            pdf.savefig(fig)

            plt.close(fig)

    fig = plt.figure(figsize=(11,8))

    plt.axis("off")

    lines = [
        "SFX V13 OMEGA SWEEP",
        "",
        "Omega FinalErr FinalResidual AvgIter HRank VRank",
        ""
    ]

    for omega, err, resid, avgiter, hrank, vrank in results:

        lines.append(
            f"{omega:.3e} "
            f"{err:.6e} "
            f"{resid:.6e} "
            f"{avgiter:.2f} "
            f"{hrank:.2f} "
            f"{vrank:.2f}"
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
