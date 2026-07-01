# test_spectral_edge_v2b.py
# Adaptive Width Spectral Edge Study

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
TARGET=0.99

x=np.linspace(0,L,N,endpoint=False)
dx=L/N
ik_full=1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel=1j*2*np.pi*np.fft.fftfreq(MID,d=dx)


def gaussian():
    return np.exp(-((x-0.25)**2)/(0.03**2))

def sinusoid(k):
    return np.sin(2*np.pi*k*x)

def broadband():
    return np.sin(2*np.pi*8*x)+0.5*np.sin(2*np.pi*16*x)+0.25*np.sin(2*np.pi*32*x)

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


def width_energy(uA,uB,W):
    left=uA[-W:]
    right=uB[:W]
    avg=0.5*(np.fft.rfft(left)+np.fft.rfft(right))
    return np.sum(np.abs(avg)**2)


def choose_width(uA,uB):
    energies=np.array([width_energy(uA,uB,W) for W in WIDTHS])
    total=np.sum(energies)+1e-30
    frac=energies/total
    cume=np.cumsum(frac)
    idx=np.searchsorted(cume,TARGET)
    return WIDTHS[min(idx,len(WIDTHS)-1)], energies


def apply_edge(uA,uB,W):
    left=uA[-W:].copy()
    right=uB[:W].copy()
    avg=0.5*(np.fft.rfft(left)+np.fft.rfft(right))

    e=np.abs(avg)**2
    c=np.cumsum(e)/(np.sum(e)+1e-30)
    k95=np.searchsorted(c,0.95)+1
    k99=np.searchsorted(c,0.99)+1

    edge=np.fft.irfft(avg,n=W)
    uA[-W:]=edge
    uB[:W]=edge
    return uA,uB,k95,k99

pdf='test_spectral_edge_v2b.pdf'

with PdfPages(pdf) as out:

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    plt.text(0.05,0.9,'Adaptive Width Spectral Edge V2b',fontsize=18)
    plt.text(0.05,0.75,f'Candidate Widths={WIDTHS}')
    plt.text(0.05,0.65,f'Target Energy={TARGET}')
    out.savefig(fig)
    plt.close(fig)

    summary=[]

    for cname,u0 in CASES.items():
        print('Running',cname)

        ref=u0.copy()
        for _ in range(NSTEPS):
            ref=rk4(fft_rhs_full,ref)

        uA=u0[:MID].copy()
        uB=u0[MID:].copy()

        width_hist=[]
        k95_hist=[]
        k99_hist=[]
        energy_hist=[]

        for _ in range(NSTEPS):
            W,energies=choose_width(uA,uB)
            uA,uB,k95,k99=apply_edge(uA,uB,W)

            uA=rk4(fft_rhs_panel,uA)
            uB=rk4(fft_rhs_panel,uB)

            width_hist.append(W)
            k95_hist.append(k95)
            k99_hist.append(k99)
            energy_hist.append(np.sum(energies))

        sol=np.concatenate([uA,uB])
        err=np.max(np.abs(sol-ref))

        summary.append([
            cname,
            err,
            np.mean(width_hist),
            np.max(width_hist),
            np.mean(k95_hist),
            np.mean(k99_hist)
        ])

        fig,ax=plt.subplots(2,2,figsize=(10,8))
        ax[0,0].plot(width_hist)
        ax[0,0].set_title('Selected Width History')

        ax[0,1].plot(k95_hist,label='k95')
        ax[0,1].plot(k99_hist,label='k99')
        ax[0,1].legend()
        ax[0,1].set_title('Spectral Complexity')

        ax[1,0].semilogy(np.maximum(energy_hist,1e-30))
        ax[1,0].set_title('Interface Energy')

        ax[1,1].axis('off')
        ax[1,1].text(
            0.05,0.9,
            f'Max Error = {err:.3e}\n\n'
            f'Avg Width = {np.mean(width_hist):.2f}\n'
            f'Max Width = {np.max(width_hist)}\n\n'
            f'Avg k95 = {np.mean(k95_hist):.2f}\n'
            f'Avg k99 = {np.mean(k99_hist):.2f}'
        )

        fig.suptitle(cname)
        fig.tight_layout()
        out.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='SUMMARY\n\n'
    for r in summary:
        txt += f'{r[0]:10s} err={r[1]:.3e} avgW={r[2]:.2f} maxW={r[3]} k95={r[4]:.2f} k99={r[5]:.2f}\n'
    plt.text(0.01,0.99,txt,va='top',family='monospace')
    out.savefig(fig)
    plt.close(fig)

print('Saved',pdf)
