import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse.linalg import gmres, LinearOperator

# ============================================================
# Configuration
# ============================================================

N = 256
L = 1.0
MID = N // 2

DT = 5e-5
NSTEPS = 2000

PDF_NAME = "burgers_gmres_interface_report.pdf"

# ============================================================
# Grid
# ============================================================

x = np.linspace(0.0, L, N, endpoint=False)
dx = L / N

# ============================================================
# Initial condition
# ============================================================

u0 = 1.0 + 0.5 * np.exp(
    -((x - 0.30) / 0.05) ** 2
)

# ============================================================
# FD operator
# ============================================================

def sbp42_matrix(n, dx):

    D = np.zeros((n, n))

    for i in range(2, n - 2):
        D[i, i - 2] =  1/(12*dx)
        D[i, i - 1] = -8/(12*dx)
        D[i, i + 1] =  8/(12*dx)
        D[i, i + 2] = -1/(12*dx)

    D[0,0] = -25/(12*dx)
    D[0,1] =   4/dx
    D[0,2] =  -3/dx
    D[0,3] =   4/(3*dx)
    D[0,4] =  -1/(4*dx)

    D[1,0] = -1/(4*dx)
    D[1,1] = -5/(6*dx)
    D[1,2] =  3/(2*dx)
    D[1,3] = -1/(2*dx)
    D[1,4] =  1/(12*dx)

    D[-1,:] = -D[0,::-1]
    D[-2,:] = -D[1,::-1]

    H = np.ones(n) * dx
    H[0] *= 0.5
    H[-1] *= 0.5

    HI = np.diag(1.0 / H)

    return H, HI, D

HA, HIA, DA = sbp42_matrix(MID, dx)
HB, HIB, DB = sbp42_matrix(MID, dx)
HF, HIF, DF = sbp42_matrix(N, dx)

eL = np.zeros(MID)
eL[0] = 1.0

eR = np.zeros(MID)
eR[-1] = 1.0

# ============================================================
# Burgers flux
# ============================================================

def flux(u):
    return 0.5 * u * u

def entropy_rusanov_flux(uL, uR):

    lam = max(abs(uL), abs(uR))

    fec = (
        uL*uL +
        uL*uR +
        uR*uR
    ) / 6.0

    return fec - 0.5 * lam * (uR - uL)

# ============================================================
# Reference
# ============================================================

def rhs_ref(u):
    return -(DF @ flux(u))

def rk4_ref(u):

    k1 = rhs_ref(u)
    k2 = rhs_ref(u + 0.5 * DT * k1)
    k3 = rhs_ref(u + 0.5 * DT * k2)
    k4 = rhs_ref(u + DT * k3)

    return u + DT * (
        k1 + 2*k2 + 2*k3 + k4
    ) / 6.0

# ============================================================
# GMRES interface solve
# ============================================================

gmres_hist = []

def solve_interface_gmres(uA, uB):

    gamma0 = np.array(
        [uA[-1], uB[0]],
        dtype=float
    )

    def matvec(v):

        return np.array([
            v[0] - v[1],
            v[1] - v[0]
        ])

    A = LinearOperator(
        (2, 2),
        matvec=matvec
    )

    rhs = np.array([
        -(gamma0[0] - gamma0[1]),
        -(gamma0[1] - gamma0[0])
    ])

    local_hist = []

    def callback(r):
        try:
            local_hist.append(float(r))
        except Exception:
            pass

    dgamma, info = gmres(
        A,
        rhs,
        restart=2,
        maxiter=10,
        atol=1e-14,
        rtol=1e-12,
        callback=callback,
        callback_type="pr_norm",
    )

    if local_hist:
        gmres_hist.append(local_hist[-1])

    gamma = gamma0 + dgamma

    uA[-1] = gamma[0]
    uB[0]  = gamma[1]

    return uA, uB

# ============================================================
# SAT RHS
# ============================================================

def rhs_sat(uA, uB):

    rA = -(DA @ flux(uA))
    rB = -(DB @ flux(uB))

    uL = uA[-1]
    uR = uB[0]

    fL = flux(uL)
    fR = flux(uR)

    fs = entropy_rusanov_flux(
        uL,
        uR
    )

    rA -= HIA @ (
        eR * (fs - fL)
    )

    rB += HIB @ (
        eL * (fs - fR)
    )

    return rA, rB

def rk4_sat(uA, uB):

    k1A, k1B = rhs_sat(uA, uB)

    k2A, k2B = rhs_sat(
        uA + 0.5*DT*k1A,
        uB + 0.5*DT*k1B
    )

    k3A, k3B = rhs_sat(
        uA + 0.5*DT*k2A,
        uB + 0.5*DT*k2B
    )

    k4A, k4B = rhs_sat(
        uA + DT*k3A,
        uB + DT*k3B
    )

    uA = uA + DT * (
        k1A + 2*k2A + 2*k3A + k4A
    ) / 6.0

    uB = uB + DT * (
        k1B + 2*k2B + 2*k3B + k4B
    ) / 6.0

    return uA, uB

# ============================================================
# Reference
# ============================================================

print("Computing reference...")

u_ref = u0.copy()

for _ in range(NSTEPS):
    u_ref = rk4_ref(u_ref)

# ============================================================
# GMRES run
# ============================================================

uA = u0[:MID].copy()
uB = u0[MID:].copy()

jump_hist = []

for _ in range(NSTEPS):

    uA, uB = solve_interface_gmres(
        uA,
        uB
    )

    jump_hist.append(
        abs(uA[-1] - uB[0])
    )

    uA, uB = rk4_sat(
        uA,
        uB
    )

u = np.concatenate(
    [uA, uB]
)

linf = np.max(
    np.abs(u - u_ref)
)

l2 = (
    np.linalg.norm(u - u_ref)
    /
    np.linalg.norm(u_ref)
)

# ============================================================
# PDF
# ============================================================

with PdfPages(PDF_NAME) as pdf:

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        x,
        u_ref,
        'k',
        lw=3,
        label='Reference'
    )

    ax.plot(
        x,
        u,
        'r--',
        label='GMRES Interface'
    )

    ax.axvline(
        x[MID],
        color='k',
        ls=':'
    )

    ax.legend()

    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.semilogy(
        np.maximum(
            jump_hist,
            1e-16
        )
    )

    ax.set_title(
        'Interface Jump'
    )

    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.semilogy(
        np.maximum(
            np.asarray(gmres_hist + [1e-16]),
            1e-16
        )
    )

    ax.set_title(
        'GMRES Residual'
    )

    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(
        figsize=(11,8)
    )

    plt.axis('off')

    summary_text = (
        f"Linf = {linf:.6e}\n"
        f"L2 = {l2:.6e}\n"
        f"Final Jump = {jump_hist[-1]:.6e}\n"
    )

    plt.text(
        0.02,
        0.95,
        summary_text,
        family='monospace',
        va='top'
    )

    pdf.savefig(fig)
    plt.close(fig)

print("Saved", PDF_NAME)
print("Linf =", linf)
print("L2 =", l2)
