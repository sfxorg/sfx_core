import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# SFX V9
# FFT vs POD Flux-Residual Schwarz
# ==========================================================

NX = 128
NY = 128
MID = NX // 2

NT = 250
N_SCHWARZ = 5

CFL = 0.35
VX = 1.0
VY = 0.5

OMEGA = 0.01
SIGMA = 0.05

TOLS = [1e-2, 1e-3, 1e-4]

PDF = "07_SFX_FFT_vs_POD_Schwarz_V9.pdf"

# ==========================================================
# Grid
# ==========================================================

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

u0 = np.exp(
    -((X-0.25)**2 + (Y-0.50)**2)
    /
    SIGMA**2
)

# ==========================================================
# Transport
# ==========================================================

def full_step(q):

    qx = q - np.roll(q,1,axis=0)
    qy = q - np.roll(q,1,axis=1)

    return q - CFL*VX*qx - CFL*VY*qy

def split(q):

    return (
        q[:MID,:].copy(),
        q[MID:,:].copy()
    )

def merge(a,b):

    return np.vstack([a,b])

# ==========================================================
# Fluxes
# ==========================================================

def physical_flux(u):

    return 0.5*u*u

def rusanov_flux(uL,uR):

    lam = np.maximum(
        np.abs(uL),
        np.abs(uR)
    )

    fL = physical_flux(uL)
    fR = physical_flux(uR)

    return (
        0.5*(fL+fR)
        -
        0.5*lam*(uR-uL)
    )

# ==========================================================
# FFT Compression
# ==========================================================

def fft_compress(trace,tol):

    c = np.fft.rfft(trace)

    e = np.abs(c)**2
    total = e.sum() + 1e-16

    run = 0.0

    rank = len(c)

    for r in range(1,len(c)+1):

        run += e[r-1]

        rem = max((total-run)/total,0.0)

        if np.sqrt(rem) <= tol:
            rank = r
            break

    cc = c.copy()
    cc[rank:] = 0

    rec = np.fft.irfft(
        cc,
        n=len(trace)
    )

    return rec,rank

# ==========================================================
# POD Training
# ==========================================================

print("Building POD basis...")

snapshots = []

q = u0.copy()

for _ in range(500):

    q = full_step(q)

    snapshots.append(
        q[MID-1,:].copy()
    )

S = np.asarray(snapshots)

mean_trace = np.mean(
    S,
    axis=0
)

A = (S - mean_trace).T

U,Sigma,VT = np.linalg.svd(
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

def pod_compress(trace,tol):

    r = pod_rank(tol)

    centered = trace - mean_trace

    rec = (
        U[:,:r]
        @
        (
            U[:,:r].T
            @
            centered
        )
        +
        mean_trace
    )

    return rec,r

# ==========================================================
# Schwarz
# ==========================================================

def schwarz_step(
    left,
    right,
    tol,
    method
):

    ranks = []
    residuals = []

    for _ in range(N_SCHWARZ):

        uL = left[-1,:]
        uR = right[0,:]

        flux = rusanov_flux(
            uL,
            uR
        )

        if method == "FFT":

            flux_rec,rank = fft_compress(
                flux,
                tol
            )

        else:

            flux_rec,rank = pod_compress(
                flux,
                tol
            )

        residual = (
            physical_flux(uL)
            -
            physical_flux(uR)
        )

        residual = np.clip(
            residual,
            -10.0,
            10.0
        )

        left[-1,:] -= (
            OMEGA*residual
        )

        right[0,:] += (
            OMEGA*residual
        )

        residuals.append(
            np.linalg.norm(residual)
        )

        ranks.append(rank)

    q = full_step(
        merge(left,right)
    )

    left,right = split(q)

    return (
        left,
        right,
        np.mean(ranks),
        residuals[-1]
    )

# ==========================================================
# Reference
# ==========================================================

ref = u0.copy()

ref_hist = []

for _ in range(NT):

    ref = full_step(ref)

    ref_hist.append(
        ref.copy()
    )

# ==========================================================
# Experiments
# ==========================================================

results = []

with PdfPages(PDF) as pdf:

    for method in ["FFT","POD"]:

        for tol in TOLS:

            print(
                method,
                "tol=",
                tol
            )

            left,right = split(
                u0.copy()
            )

            errs = []
            ranks = []
            residuals = []

            for n in range(NT):

                (
                    left,
                    right,
                    rank,
                    resid
                ) = schwarz_step(
                    left,
                    right,
                    tol,
                    method
                )

                q = merge(
                    left,
                    right
                )

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
                ranks.append(rank)
                residuals.append(resid)

            results.append(
                (
                    method,
                    tol,
                    errs[-1],
                    np.mean(ranks),
                    residuals[-1]
                )
            )

            fig,ax = plt.subplots(
                1,
                3,
                figsize=(12,4)
            )

            ax[0].semilogy(
                np.maximum(
                    errs,
                    1e-16
                )
            )
            ax[0].set_title("L2 Error")

            ax[1].plot(ranks)
            ax[1].set_title("Rank")

            ax[2].semilogy(
                np.maximum(
                    residuals,
                    1e-16
                )
            )
            ax[2].set_title(
                "Residual"
            )

            fig.suptitle(
                f"{method}  tol={tol:.0e}"
            )

            pdf.savefig(fig)
            plt.close(fig)

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    lines = [
        "SFX V9 FFT vs POD",
        "",
        "Method Tol FinalErr AvgRank FinalResidual",
        ""
    ]

    for m,t,e,r,res in results:

        lines.append(
            f"{m:4s} "
            f"{t:.0e} "
            f"{e:.6e} "
            f"{r:.2f} "
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

for row in results:
    print(row)
