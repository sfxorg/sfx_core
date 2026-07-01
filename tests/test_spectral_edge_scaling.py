# test_spectral_edge_scaling.py
# Scaling-Law Study for Spectral Edge Interfaces

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N = 256
L = 1.0
MID = N // 2
DT = 2e-4
NSTEPS = 500
C = 1.0

WIDTHS = [2,4,8,16,32,64]
RANKS = [1,2,4,8,16]
FREQS = [2,4,8,16,32,64]

x = np.linspace(0,L,N,endpoint=False)
dx = L/N

ik_full = 1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel = 1j*2*np.pi*np.fft.fftfreq(MID,d=dx)


def fft_rhs_full(u):
    return -C*np.fft.ifft(ik_full*np.fft.fft(u)).real


def fft_rhs_panel(u):
    return -C*np.fft.ifft(ik_panel*np.fft.fft(u)).real


def rk4(rhs,u):
    k1=rhs(u)
    k2=rhs(u+0.5*DT*k1)
    k3=rhs(u+0.5*DT*k2)
    k4=rhs(u+DT*k3)
    return u + DT*(k1+2*k2+2*k3+k4)/6.0


def sinusoid(k):
    return np.sin(2*np.pi*k*x)


def spectral_edge(uA,uB,W,K):

    left=uA[-W:].copy()
    right=uB[:W].copy()

    spec=0.5*(np.fft.rfft(left)+np.fft.rfft(right))

    e=np.abs(spec)**2
    c=np.cumsum(e)/(e.sum()+1e-30)

    k95=np.searchsorted(c,0.95)+1

    trunc=spec.copy()
    trunc[min(K,len(trunc)):]=0.0

    edge=np.fft.irfft(trunc,n=W)

    uA[-W:]=edge
    uB[:W]=edge

    return uA,uB,k95


pdf_name='test_spectral_edge_scaling.pdf'

summary=[]

with PdfPages(pdf_name) as pdf:

    for freq in FREQS:

        print('Running frequency',freq)

        u0=sinusoid(freq)

        ref=u0.copy()
        for _ in range(NSTEPS):
            ref=rk4(fft_rhs_full,ref)

        err_map=np.zeros((len(WIDTHS),len(RANKS)))

        for i,W in enumerate(WIDTHS):
            for j,K in enumerate(RANKS):

                uA=u0[:MID].copy()
                uB=u0[MID:].copy()

                for _ in range(NSTEPS):
                    uA,uB,_=spectral_edge(uA,uB,W,K)
                    uA=rk4(fft_rhs_panel,uA)
                    uB=rk4(fft_rhs_panel,uB)

                sol=np.concatenate([uA,uB])
                err_map[i,j]=np.max(np.abs(sol-ref))

        best=np.unravel_index(np.argmin(err_map),err_map.shape)

        bestW=WIDTHS[best[0]]
        bestK=RANKS[best[1]]
        bestErr=float(err_map[best])

        wavelength=L/freq
        compression=N/(bestW*bestK)

        summary.append([
            freq,
            wavelength,
            bestW,
            bestK,
            compression,
            bestErr
        ])

        fig,ax=plt.subplots(figsize=(7,5))
        im=ax.imshow(
            np.log10(np.maximum(err_map,1e-16)),
            origin='lower',
            aspect='auto'
        )
        ax.set_xticks(range(len(RANKS)))
        ax.set_xticklabels(RANKS)
        ax.set_yticks(range(len(WIDTHS)))
        ax.set_yticklabels(WIDTHS)
        ax.set_xlabel('Rank')
        ax.set_ylabel('Width')
        ax.set_title(
            f'Sin{freq}  Best(W={bestW},K={bestK})'
        )
        plt.colorbar(im,ax=ax,label='log10(error)')
        pdf.savefig(fig)
        plt.close(fig)

    freqs=np.array([s[0] for s in summary])
    widths=np.array([s[2] for s in summary])
    ranks=np.array([s[3] for s in summary])
    comp=np.array([s[4] for s in summary])

    fig,ax=plt.subplots(1,3,figsize=(12,4))

    ax[0].plot(freqs,widths,'o-')
    ax[0].set_title('Best Width vs Frequency')
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('Best Width')

    ax[1].plot(freqs,ranks,'o-')
    ax[1].set_title('Best Rank vs Frequency')
    ax[1].set_xlabel('Frequency')
    ax[1].set_ylabel('Best Rank')

    ax[2].plot(freqs,comp,'o-')
    ax[2].set_title('Compression Ratio')
    ax[2].set_xlabel('Frequency')
    ax[2].set_ylabel('N/(W*K)')

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='SCALING LAW SUMMARY\n\n'
    txt+='Freq  Lambda    W*   K*   Compression    Error\n\n'

    for s in summary:
        txt += f'{s[0]:4d}  {s[1]:.5f}  {s[2]:4d}  {s[3]:4d}  {s[4]:10.2f}  {s[5]:.3e}\n'

    plt.text(0.01,0.99,txt,va='top',family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

print('Saved',pdf_name)
