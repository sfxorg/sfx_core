# test_spectral_edge_v3.py
# Width x Rank Phase Diagram Study

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N=256
L=1.0
MID=N//2
DT=2e-4
NSTEPS=500
C=1.0

WIDTHS=[2,4,8,16,32]
RANKS=[1,2,4,8,16]

x=np.linspace(0,L,N,endpoint=False)
dx=L/N
ik_full=1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel=1j*2*np.pi*np.fft.fftfreq(MID,d=dx)


def gaussian():
    return np.exp(-((x-0.25)**2)/(0.03**2))


def sinusoid(k):
    return np.sin(2*np.pi*k*x)


def broadband():
    return (
        np.sin(2*np.pi*8*x)
        +0.5*np.sin(2*np.pi*16*x)
        +0.25*np.sin(2*np.pi*32*x)
    )

CASES={
 'Gaussian':gaussian(),
 'Sin8':sinusoid(8),
 'Sin32':sinusoid(32),
 'Broadband':broadband(),
}


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


def spectral_edge(uA,uB,W,K):
    left=uA[-W:].copy()
    right=uB[:W].copy()

    spec=0.5*(np.fft.rfft(left)+np.fft.rfft(right))

    e=np.abs(spec)**2
    c=np.cumsum(e)/(np.sum(e)+1e-30)

    k95=np.searchsorted(c,0.95)+1
    k99=np.searchsorted(c,0.99)+1

    trunc=spec.copy()
    trunc[min(K,len(trunc)):]=0

    edge=np.fft.irfft(trunc,n=W)

    uA[-W:]=edge
    uB[:W]=edge

    return uA,uB,k95,k99

pdf='test_spectral_edge_v3.pdf'

with PdfPages(pdf) as out:

    summary=[]

    for cname,u0 in CASES.items():

        print('Running',cname)

        ref=u0.copy()
        for _ in range(NSTEPS):
            ref=rk4(fft_rhs_full,ref)

        err_map=np.zeros((len(WIDTHS),len(RANKS)))
        k95_map=np.zeros_like(err_map)

        for i,W in enumerate(WIDTHS):
            for j,K in enumerate(RANKS):

                uA=u0[:MID].copy()
                uB=u0[MID:].copy()

                k95_hist=[]

                for _ in range(NSTEPS):
                    uA,uB,k95,k99=spectral_edge(uA,uB,W,K)
                    uA=rk4(fft_rhs_panel,uA)
                    uB=rk4(fft_rhs_panel,uB)
                    k95_hist.append(k95)

                sol=np.concatenate([uA,uB])

                err=np.max(np.abs(sol-ref))

                err_map[i,j]=err
                k95_map[i,j]=np.mean(k95_hist)

        best=np.unravel_index(np.argmin(err_map),err_map.shape)
        bestW=WIDTHS[best[0]]
        bestK=RANKS[best[1]]
        bestErr=err_map[best]

        summary.append((cname,bestW,bestK,bestErr))

        fig,ax=plt.subplots(1,2,figsize=(11,5))

        im=ax[0].imshow(
            np.log10(np.maximum(err_map,1e-16)),
            aspect='auto',origin='lower'
        )
        ax[0].set_title('log10(Max Error)')
        ax[0].set_xticks(range(len(RANKS)))
        ax[0].set_xticklabels(RANKS)
        ax[0].set_yticks(range(len(WIDTHS)))
        ax[0].set_yticklabels(WIDTHS)
        ax[0].set_xlabel('Rank')
        ax[0].set_ylabel('Width')
        plt.colorbar(im,ax=ax[0])

        im2=ax[1].imshow(
            k95_map,
            aspect='auto',origin='lower'
        )
        ax[1].set_title('Average k95')
        ax[1].set_xticks(range(len(RANKS)))
        ax[1].set_xticklabels(RANKS)
        ax[1].set_yticks(range(len(WIDTHS)))
        ax[1].set_yticklabels(WIDTHS)
        ax[1].set_xlabel('Rank')
        ax[1].set_ylabel('Width')
        plt.colorbar(im2,ax=ax[1])

        fig.suptitle(
            f'{cname}\nBest: W={bestW}, K={bestK}, err={bestErr:.3e}'
        )
        fig.tight_layout()
        out.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='WIDTH-RANK PHASE DIAGRAM SUMMARY\n\n'
    for s in summary:
        txt += f'{s[0]:10s} bestW={s[1]:2d} bestK={s[2]:2d} err={s[3]:.3e}\n'
    plt.text(0.01,0.99,txt,va='top',family='monospace')
    out.savefig(fig)
    plt.close(fig)

print('Saved',pdf)
