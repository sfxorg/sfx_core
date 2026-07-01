import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =====================================================
# CONFIG
# =====================================================

NX = 128
NY = 128

MID = NX // 2

NT = 250

CFL = 0.35

VX = 1.0
VY = 0.5

SIGMA = 0.05

RANKS = [1,2,4,8,16,32,64]

# =====================================================
# GRID
# =====================================================

x = np.linspace(
    0,
    1,
    NX,
    endpoint=False
)

y = np.linspace(
    0,
    1,
    NY,
    endpoint=False
)

X,Y = np.meshgrid(
    x,
    y,
    indexing="ij"
)

# =====================================================
# TEST CASES
# =====================================================

def gaussian():

    return np.exp(
        -(
            (X-0.25)**2 +
            (Y-0.50)**2
        )/SIGMA**2
    )

def sinusoid(k):

    return (
        gaussian()
        *
        np.sin(
            2*np.pi*k*X
        )
    )

def broadband():

    env = gaussian()

    return env * (
        np.sin(2*np.pi*8*X)
        +
        0.7*np.sin(2*np.pi*16*X)
        +
        0.5*np.sin(2*np.pi*32*X)
        +
        0.3*np.sin(2*np.pi*64*X)
    )

def random_packet():

    rng = np.random.default_rng(1)

    q = np.zeros((NX,NY))

    for k in [4,8,16,24,32,48]:

        amp = rng.uniform(0.2,1.0)

        phase = rng.uniform(
            0,
            2*np.pi
        )

        q += (
            amp
            *
            np.sin(
                2*np.pi*k*X
                +
                phase
            )
        )

    return q * gaussian()

CASES = {
    "Gaussian" : gaussian(),
    "Sin8"     : sinusoid(8),
    "Sin32"    : sinusoid(32),
    "Broadband": broadband(),
    "Random"   : random_packet()
}

# =====================================================
# MONOLITHIC SOLVER
# =====================================================

def full_step(q):

    qx = q - np.roll(
        q,
        1,
        axis=0
    )

    qy = q - np.roll(
        q,
        1,
        axis=1
    )

    return (
        q
        -
        CFL * VX * qx
        -
        CFL * VY * qy
    )

# =====================================================
# SPLIT / MERGE
# =====================================================

def split(q):

    return (
        q[:MID,:].copy(),
        q[MID:,:].copy()
    )

def merge(left,right):

    return np.vstack(
        [left,right]
    )

# =====================================================
# COMMUNICATION OPERATORS
# =====================================================

def exact_transfer(trace):

    return trace.copy()

def fft_transfer(trace,rank):

    coeff = np.fft.rfft(trace)

    coeff[rank:] = 0

    return np.fft.irfft(
        coeff,
        n=len(trace)
    )

def build_pod_basis(A):

    U,S,VT = np.linalg.svd(
        A,
        full_matrices=False
    )

    return U,S,VT

def pod_transfer(
    trace,
    U,
    rank
):

    coeff = U[:,:rank].T @ trace

    return U[:,:rank] @ coeff

# =====================================================
# COMMUNICATED STEP
# =====================================================

def communicated_step(
    left,
    right,
    method,
    rank,
    POD_U=None
):

    send_LR = left[-1,:]
    send_RL = right[-1,:]

    if method == "EXACT":

        ghost_right = send_LR.copy()
        ghost_left  = send_RL.copy()

    elif method == "FFT":

        ghost_right = fft_transfer(
            send_LR,
            rank
        )

        ghost_left = fft_transfer(
            send_RL,
            rank
        )

    else:

        ghost_right = pod_transfer(
            send_LR,
            POD_U,
            rank
        )

        ghost_left = pod_transfer(
            send_RL,
            POD_U,
            rank
        )

    q = merge(left,right)

    q[MID-1,:] = ghost_right
    q[-1,:]    = ghost_left

    qnew = full_step(q)

    return split(qnew)

# =====================================================
# POD TRAINING
# =====================================================

ref = gaussian()

snaps = []

for _ in range(500):

    ref = full_step(ref)

    snaps.append(
        ref[MID-1,:]
    )

A = np.asarray(snaps).T

POD_U,S,VT = build_pod_basis(A)

# =====================================================
# PDF
# =====================================================

pdf_name = (
    "02_Compression_Suite_CORRECTED.pdf"
)

with PdfPages(pdf_name) as pdf:

    summary = []

    for case,u0 in CASES.items():

        print(case)

        ref = u0.copy()

        ref_hist = []

        for _ in range(NT):

            ref = full_step(ref)

            ref_hist.append(ref.copy())

        for method in [
            "EXACT",
            "FFT",
            "POD"
        ]:

            ranks = [64] if method=="EXACT" else RANKS

            for rank in ranks:

                left,right = split(
                    u0.copy()
                )

                l2_hist = []

                for n in range(NT):

                    left,right = (
                        communicated_step(
                            left,
                            right,
                            method,
                            rank,
                            POD_U
                        )
                    )

                    q = merge(
                        left,
                        right
                    )

                    ref = ref_hist[n]

                    err = (
                        np.linalg.norm(
                            q-ref
                        )
                        /
                        (
                            np.linalg.norm(ref)
                            +1e-16
                        )
                    )

                    l2_hist.append(err)

                final_error = l2_hist[-1]

                summary.append(
                    [
                        case,
                        method,
                        rank,
                        final_error
                    ]
                )

                fig,axs = plt.subplots(
                    1,
                    3,
                    figsize=(14,4)
                )

                axs[0].imshow(
                    ref.T,
                    origin="lower"
                )

                axs[0].set_title(
                    "Reference"
                )

                axs[1].imshow(
                    q.T,
                    origin="lower"
                )

                axs[1].set_title(
                    f"{method} r={rank}"
                )

                axs[2].semilogy(
                    l2_hist
                )

                axs[2].set_title(
                    f"L2={final_error:.3e}"
                )

                fig.suptitle(
                    f"{case}"
                )

                fig.tight_layout()

                pdf.savefig(fig)

                plt.close(fig)

    summary = np.array(
        summary,
        dtype=object
    )

    fig,ax = plt.subplots(
        figsize=(8,5)
    )

    methods = summary[:,1]

    ranks = summary[:,2].astype(float)

    errors = summary[:,3].astype(float)

    for m in np.unique(methods):

        mask = methods == m

        ax.scatter(
            ranks[mask],
            errors[mask],
            label=m
        )

    ax.set_yscale("log")

    ax.set_xlabel(
        "Rank"
    )

    ax.set_ylabel(
        "Final Error"
    )

    ax.legend()

    ax.set_title(
        "Compression Study"
    )

    pdf.savefig(fig)

    plt.close(fig)

print()
print("Saved:",pdf_name)
print()
