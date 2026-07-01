# test_spectral_edge_v2a.py
# Clean Width Study for Virtual Spectral Edge

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================================
# CONFIG
# ==========================================================

N = 256
L = 1.0
MID = N // 2

DT = 2e-4
NSTEPS = 500
C = 1.0

WIDTHS = [2, 4, 8, 16, 32]
K = 4

# ==========================================================
# GRID
# ==========================================================

x = np.linspace(0.0, L, N, endpoint=False)
dx = L / N

ik_full = 1j * 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ik_panel = 1j * 2.0 * np.pi * np.fft.fftfreq(MID, d=dx)

# ==========================================================
# TEST CASES
# ==========================================================

def gaussian():
    return np.exp(-((x - 0.25) ** 2) / (0.03 ** 2))


def sinusoid(k):
    return np.sin(2.0 * np.pi * k * x)


def broadband():
    return (
        np.sin(2 * np.pi * 8 * x)
        + 0.5 * np.sin(2 * np.pi * 16 * x)
        + 0.25 * np.sin(2 * np.pi * 32 * x)
    )


CASES = {
    'Gaussian': gaussian(),
    'Sin8': sinusoid(8),
    'Sin32': sinusoid(32),
    'Broadband': broadband(),
}

# ==========================================================
# SOLVERS
# ==========================================================

def fft_rhs_full(u):
    uhat = np.fft.fft(u)
    ux = np.fft.ifft(ik_full * uhat).real
    return -C * ux


def fft_rhs_panel(u):
    uhat = np.fft.fft(u)
    ux = np.fft.ifft(ik_panel * uhat).real
    return -C * ux


def rk4(rhs, u):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * DT * k1)
    k3 = rhs(u + 0.5 * DT * k2)
    k4 = rhs(u + DT * k3)

    return u + DT * (
        k1 + 2 * k2 + 2 * k3 + k4
    ) / 6.0

# ==========================================================
# SPECTRAL EDGE COUPLING
# ==========================================================

def spectral_edge(uA, uB, width, rank):

    left = uA[-width:].copy()
    right = uB[:width].copy()

    LA = np.fft.rfft(left)
    RB = np.fft.rfft(right)

    avg = 0.5 * (LA + RB)

    energy = np.abs(avg) ** 2

    total = energy.sum() + 1e-30

    cume = np.cumsum(energy) / total

    k95 = np.searchsorted(cume, 0.95) + 1
    k99 = np.searchsorted(cume, 0.99) + 1

    avg_trunc = avg.copy()
    avg_trunc[rank:] = 0.0

    edge = np.fft.irfft(avg_trunc, n=width)

    uA[-width:] = edge
    uB[:width] = edge

    return uA, uB, k95, k99, total

# ==========================================================
# PDF REPORT
# ==========================================================

pdf_name = 'test_spectral_edge_v2a.pdf'
summary = []

with PdfPages(pdf_name) as pdf:

    fig = plt.figure(figsize=(11, 8))
    plt.axis('off')
    plt.text(0.05, 0.90, 'Spectral Edge Width Study V2a', fontsize=18)
    plt.text(0.05, 0.75, f'Widths = {WIDTHS}')
    plt.text(0.05, 0.65, f'Rank = {K}')
    plt.text(0.05, 0.55, f'NSTEPS = {NSTEPS}')
    pdf.savefig(fig)
    plt.close(fig)

    for cname, u0 in CASES.items():

        print(f'Running {cname}')

        u_ref = u0.copy()

        for _ in range(NSTEPS):
            u_ref = rk4(fft_rhs_full, u_ref)

        widths_out = []
        errs_out = []
        k95_out = []
        k99_out = []
        ener_out = []

        for W in WIDTHS:

            uA = u0[:MID].copy()
            uB = u0[MID:].copy()

            k95_hist = []
            k99_hist = []
            energy_hist = []

            for _ in range(NSTEPS):

                uA, uB, k95, k99, energy = spectral_edge(
                    uA, uB, W, K
                )

                uA = rk4(fft_rhs_panel, uA)
                uB = rk4(fft_rhs_panel, uB)

                k95_hist.append(k95)
                k99_hist.append(k99)
                energy_hist.append(energy)

            u_two = np.concatenate([uA, uB])

            maxerr = np.max(np.abs(u_ref - u_two))

            widths_out.append(W)
            errs_out.append(maxerr)
            k95_out.append(np.mean(k95_hist))
            k99_out.append(np.mean(k99_hist))
            ener_out.append(np.mean(energy_hist))

            summary.append(
                [
                    cname,
                    W,
                    maxerr,
                    np.mean(k95_hist),
                    np.mean(k99_hist),
                ]
            )

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].plot(widths_out, errs_out, '-o')
        ax[0, 0].set_yscale('log')
        ax[0, 0].set_title('Max Error vs Width')

        ax[0, 1].plot(widths_out, k95_out, '-o')
        ax[0, 1].set_title('Average k95')

        ax[1, 0].plot(widths_out, k99_out, '-o')
        ax[1, 0].set_title('Average k99')

        ax[1, 1].plot(widths_out, ener_out, '-o')
        ax[1, 1].set_yscale('log')
        ax[1, 1].set_title('Average Edge Energy')

        fig.suptitle(cname)
        fig.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(figsize=(11, 8))
    plt.axis('off')

    txt = 'SUMMARY\n\n'

    for row in summary:
        txt += (
            f'{row[0]:10s} '
            f'W={row[1]:2d} '
            f'err={row[2]:.3e} '
            f'k95={row[3]:.2f} '
            f'k99={row[4]:.2f}\n'
        )

    plt.text(
        0.01,
        0.99,
        txt,
        va='top',
        family='monospace'
    )

    pdf.savefig(fig)
    plt.close(fig)

print('Saved', pdf_name)
