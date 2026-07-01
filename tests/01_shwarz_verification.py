import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

# =====================================================
# CONFIG
# =====================================================

NX = 128
NY = 128

LX = 1.0
LY = 1.0

MID = NX // 2

NT = 250

CFL = 0.35

VX = 1.0
VY = 0.5

SIGMA = 0.05

# =====================================================
# GRID
# =====================================================

x = np.linspace(
    0,
    LX,
    NX,
    endpoint=False
)

y = np.linspace(
    0,
    LY,
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

def constant_field():

    return np.ones((NX,NY))


def gaussian():

    return np.exp(
        -(
            (X-0.25)**2
            +
            (Y-0.50)**2
        ) / SIGMA**2
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

        amp = rng.uniform(
            0.2,
            1.0
        )

        phase = rng.uniform(
            0,
            2*np.pi
        )

        q += amp*np.sin(
            2*np.pi*k*X
            +
            phase
        )

    return q * gaussian()


def oblique(k,angle):

    theta = np.deg2rad(angle)

    phase = (
        np.cos(theta)*X
        +
        np.sin(theta)*Y
    )

    return (
        gaussian()
        *
        np.sin(
            2*np.pi*k*phase
        )
    )

# =====================================================
# CASES
# =====================================================

CASES = {

    "Constant":
        constant_field(),

    "Gaussian":
        gaussian(),

    "Sin8":
        sinusoid(8),

    "Sin32":
        sinusoid(32),

    "Broadband":
        broadband(),

    "Random":
        random_packet(),

    "Oblique15":
        oblique(32,15),

    "Oblique30":
        oblique(32,30),

    "Oblique45":
        oblique(32,45),

    "Oblique60":
        oblique(32,60)
}

# =====================================================
# MONOLITHIC SOLVER
# =====================================================

def monolithic_step(q):

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
        CFL*VX*qx
        -
        CFL*VY*qy
    )

# =====================================================
# PANEL UTILITIES
# =====================================================

def split_domain(q):

    left = q[:MID,:].copy()
    right = q[MID:,:].copy()

    return left,right


def merge_domain(left,right):

    return np.vstack(
        [left,right]
    )

# =====================================================
# EXACT DECOMPOSED STEP
# =====================================================

def decomposed_step(left,right):

    q = merge_domain(
        left,
        right
    )

    q = monolithic_step(q)

    left,right = split_domain(q)

    return left,right

# =====================================================
# SINGLE STEP STENCIL TEST
# =====================================================

print()
print("========================================")
print("SINGLE STEP VERIFICATION")
print("========================================")

u0 = np.random.default_rng(0).random(
    (NX,NY)
)

q_ref = monolithic_step(
    u0.copy()
)

left,right = split_domain(
    u0.copy()
)

left,right = decomposed_step(
    left,
    right
)

q_test = merge_domain(
    left,
    right
)

maxerr = np.max(
    np.abs(
        q_ref-q_test
    )
)

print(
    "Single-step max error =",
    maxerr
)

print()

# =====================================================
# PDF
# =====================================================

pdf_name = "01_Verification_Suite_FIXED.pdf"

summary = []

with PdfPages(pdf_name) as pdf:

    for name,u0 in CASES.items():

        print("Running:",name)

        # ---------------------------------
        # REFERENCE RUN
        # ---------------------------------

        q_ref = u0.copy()

        ref_history = []

        for _ in range(NT):

            q_ref = monolithic_step(
                q_ref
            )

            ref_history.append(
                q_ref.copy()
            )

        # ---------------------------------
        # DECOMPOSED RUN
        # ---------------------------------

        left,right = split_domain(
            u0.copy()
        )

        l2_hist = []
        linf_hist = []

        mass_hist = []

        energy_hist = []

        jump_hist = []

        for n in range(NT):

            left,right = decomposed_step(
                left,
                right
            )

            q_panel = merge_domain(
                left,
                right
            )

            q_ref_now = ref_history[n]

            diff = (
                q_panel
                -
                q_ref_now
            )

            l2 = (
                np.linalg.norm(diff)
                /
                (
                    np.linalg.norm(
                        q_ref_now
                    )
                    + 1e-16
                )
            )

            linf = np.max(
                np.abs(diff)
            )

            mass_error = abs(
                np.sum(q_panel)
                -
                np.sum(q_ref_now)
            )

            energy_error = abs(
                np.sum(q_panel**2)
                -
                np.sum(q_ref_now**2)
            )

            jump = np.max(
                np.abs(
                    left[-1,:]
                    -
                    right[0,:]
                )
            )

            l2_hist.append(l2)

            linf_hist.append(linf)

            mass_hist.append(
                mass_error
            )

            energy_hist.append(
                energy_error
            )

            jump_hist.append(jump)

        q_final = q_panel
        ref_final = ref_history[-1]

        final_l2 = l2_hist[-1]
        final_linf = linf_hist[-1]

        status = (
            "PASS"
            if final_l2 < 1e-10
            else "FAIL"
        )

        summary.append(
            [
                name,
                final_l2,
                final_linf
            ]
        )

        # ---------------------------------
        # PAGE
        # ---------------------------------

        fig,axs = plt.subplots(
            2,
            3,
            figsize=(16,9)
        )

        axs[0,0].imshow(
            ref_final.T,
            origin="lower"
        )

        axs[0,0].set_title(
            "Reference"
        )

        axs[0,1].imshow(
            q_final.T,
            origin="lower"
        )

        axs[0,1].set_title(
            "Decomposed"
        )

        axs[0,2].imshow(
            (
                q_final-ref_final
            ).T,
            origin="lower"
        )

        axs[0,2].set_title(
            "Difference"
        )

        axs[1,0].semilogy(
            l2_hist,
            label="L2"
        )

        axs[1,0].semilogy(
            linf_hist,
            label="Linf"
        )

        axs[1,0].legend()

        axs[1,0].set_title(
            "Error History"
        )

        axs[1,1].plot(
            jump_hist
        )

        axs[1,1].set_title(
            "Interface Jump"
        )

        axs[1,2].plot(
            mass_hist,
            label="Mass"
        )

        axs[1,2].plot(
            energy_hist,
            label="Energy"
        )

        axs[1,2].legend()

        axs[1,2].set_title(
            "Conservation Errors"
        )

        fig.suptitle(

            f"{name} | "
            f"L2={final_l2:.3e} | "
            f"Linf={final_linf:.3e} | "
            f"{status}"
        )

        fig.tight_layout()

        pdf.savefig(fig)

        plt.close(fig)

    # =================================================
    # SUMMARY PAGE
    # =================================================

    summary = np.array(
        summary,
        dtype=object
    )

    fig,axs = plt.subplots(
        1,
        2,
        figsize=(12,5)
    )

    axs[0].bar(
        summary[:,0],
        summary[:,1].astype(float)
    )

    axs[0].set_yscale("log")

    axs[0].tick_params(
        axis="x",
        rotation=45
    )

    axs[0].set_title(
        "Final L2 Error"
    )

    axs[1].bar(
        summary[:,0],
        summary[:,2].astype(float)
    )

    axs[1].set_yscale("log")

    axs[1].tick_params(
        axis="x",
        rotation=45
    )

    axs[1].set_title(
        "Final Linf Error"
    )

    fig.tight_layout()

    pdf.savefig(fig)

    plt.close(fig)

print()
print("Saved:",pdf_name)
print()
