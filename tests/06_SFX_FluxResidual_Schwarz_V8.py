import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# SFX Flux-Residual Schwarz V8
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

PDF = "06_SFX_FluxResidual_Schwarz_V8.pdf"

x = np.linspace(0,1,NX,endpoint=False)
y = np.linspace(0,1,NY,endpoint=False)

X,Y = np.meshgrid(x,y,indexing="ij")

u0 = np.exp(
    -((X-0.25)**2 + (Y-0.5)**2)
    / SIGMA**2
)

# ==========================================================
# Solver
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

def merge(left,right):

    return np.vstack([left,right])

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

    tot = e.sum() + 1e-16

    running = 0.0

    rank = len(c)

    for r in range(1,len(c)+1):

        running += e[r-1]

        frac = max(
            (tot-running)/tot,
            0.0
        )

        if np.sqrt(frac) <= tol:
            rank = r
            break

    cc = c.copy()
    cc[rank:] = 0

    rec = np.fft.irfft(
        cc,
        n=len(trace)
    )

    return rec, rank

# ==========================================================
# Schwarz
# ==========================================================

def schwarz_iteration(
    left,
    right,
    tol
):

    rank_hist = []
    residual_hist = []

    for _ in range(N_SCHWARZ):

        uL = left[-1,:]
        uR = right[0,:]

        flux = rusanov_flux(
            uL,
            uR
        )

        flux_rec, rank = fft_compress(
            flux,
            tol
        )

        fL = physical_flux(uL)
        fR = physical_flux(uR)

        residual = fL - fR

        residual = np.clip(
            residual,
            -10.0,
            10.0
        )

        left[-1,:] -= (
            OMEGA * residual
        )

        right[0,:] += (
            OMEGA * residual
        )

        left[-1,:] = np.clip(
            left[-1,:],
            -5,
            5
        )

        right[0,:] = np.clip(
            right[0,:],
            -5,
            5
        )

        residual_hist.append(
            np.linalg.norm(residual)
        )

        rank_hist.append(rank)

    q = full_step(
        merge(left,right)
    )

    left,right = split(q)

    return (
        left,
        right,
        max(rank_hist),
        residual_hist[-1]
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

    for tol in TOLS:

        left,right = split(
            u0.copy()
        )

        errs = []
        ranks = []
        residuals = []

        for n in range(NT):

            left,right,r,res = (
                schwarz_iteration(
                    left,
                    right,
                    tol
                )
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
            ranks.append(r)
            residuals.append(res)

        results.append(
            (
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
        ax[0].set_title(
            "L2 Error"
        )

        ax[1].plot(ranks)
        ax[1].set_title(
            "FFT Rank"
        )

        ax[2].semilogy(
            np.maximum(
                residuals,
                1e-16
            )
        )
        ax[2].set_title(
            "Flux Residual"
        )

        fig.suptitle(
            f"Flux Residual Schwarz tol={tol:.0e}"
        )

        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis("off")

    lines = [
        "SFX V8 FLUX-RESIDUAL SCHWARZ",
        "",
        "Tol FinalErr AvgRank FinalResidual",
        ""
    ]

    for t,e,r,res in results:

        lines.append(
            f"{t:.0e} "
            f"{e:.6e} "
            f"{r:.2f} "
            f"{res:.6e}"
        )

    plt.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        family="monospace"
    )

    pdf.savefig(fig)
    plt.close(fig)

print("Saved", PDF)
print("Results:")
for r in results:
    print(r)
