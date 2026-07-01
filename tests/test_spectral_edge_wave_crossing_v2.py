# test_spectral_edge_wave_crossing_v2.py
# Forced Wave-Crossing Benchmark

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N=256
L=1.0
MID=N//2
DT=2e-4
NSTEPS=2000
C=1.0

WIDTHS=[2,4,8,16,32]
RANKS=[1,2,4,8,16]

x=np.linspace(0,L,N,endpoint=False)
dx=L/N
ik_full=1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel=1j*2*np.pi*np.fft.fftfreq(MID,d=dx)

# packet starts close to interface so crossing must occur
x0=0.40
sigma=0.03
u0=np.exp(-((x-x0)**2)/(sigma**2))
u0[MID:]=0.0


def rhs_full(u):
    return -C*np.fft.ifft(ik_full*np.fft.fft(u)).real


def rhs_panel(u):
    return -C*np.fft.ifft(ik_panel*np.fft.fft(u)).real


def rk4(rhs,u):
    k1=rhs(u)
    k2=rhs(u+0.5*DT*k1)
    k3=rhs(u+0.5*DT*k2)
    k4=rhs(u+DT*k3)
    return u + DT*(k1+2*k2+2*k3+k4)/6.0


def spectral_edge(uA,uB,W,K):
    spec=0.5*(np.fft.rfft(uA[-W:])+np.fft.rfft(uB[:W]))
    e=np.abs(spec)**2
    c=np.cumsum(e)/(e.sum()+1e-30)
    k95=np.searchsorted(c,0.95)+1

    trunc=spec.copy()
    trunc[min(K,len(trunc)):]=0
    edge=np.fft.irfft(trunc,n=W)

    uA=uA.copy(); uB=uB.copy()
    uA[-W:]=edge
    uB[:W]=edge
    return uA,uB,k95

pdf_name='test_spectral_edge_wave_crossing_v2.pdf'

ref=u0.copy()
ref_hist=[]
for _ in range(NSTEPS):
    ref=rk4(rhs_full,ref)
    ref_hist.append(ref.copy())

summary=[]

with PdfPages(pdf_name) as pdf:

    for W in WIDTHS:
        for K in RANKS:
            uA=u0[:MID].copy()
            uB=u0[MID:].copy()

            arrival=None
            k95_hist=[]

            for step in range(NSTEPS):
                uA,uB,k95=spectral_edge(uA,uB,W,K)
                uA=rk4(rhs_panel,uA)
                uB=rk4(rhs_panel,uB)

                energyB=np.linalg.norm(uB)
                if arrival is None and energyB > 1e-3:
                    arrival=step*DT

                k95_hist.append(k95)

            err=np.max(np.abs(np.concatenate([uA,uB])-ref_hist[-1]))
            if arrival is None:
                arrival=-1

            summary.append((W,K,W*K,err,arrival,float(np.mean(k95_hist))))

    best=min(summary,key=lambda s:(s[2],s[3]) if s[4] >= 0 else (10**9,s[3]))

    W,K=best[0],best[1]

    uA=u0[:MID].copy(); uB=u0[MID:].copy()
    energy=[]; k95s=[]
    snapshots=[]
    snap_steps=[0,500,1000,1500,1999]

    for step in range(NSTEPS):
        uA,uB,k95=spectral_edge(uA,uB,W,K)
        uA=rk4(rhs_panel,uA)
        uB=rk4(rhs_panel,uB)

        energy.append(np.linalg.norm(uB))
        k95s.append(k95)

        if step in snap_steps:
            snapshots.append(np.concatenate([uA,uB]).copy())

    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(np.arange(NSTEPS)*DT,energy)
    ax[0].set_title('Panel B Energy vs Time')
    ax[1].plot(k95s)
    ax[1].set_title('Interface k95')
    pdf.savefig(fig)
    plt.close(fig)

    fig,axs=plt.subplots(len(snapshots),1,figsize=(8,10))
    for i,s in enumerate(snapshots):
        axs[i].plot(x,s)
        axs[i].axvline(x[MID],ls='--')
        axs[i].set_title(f'Step {snap_steps[i]}')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='WAVE CROSSING V2\n\nW K Cost Error Arrival AvgK95\n\n'
    for r in sorted(summary,key=lambda z:(z[2],z[3])):
        txt+=f'{r[0]:2d} {r[1]:2d} {r[2]:4d} {r[3]:.3e} {r[4]:.4f} {r[5]:.2f}\n'
    txt+='\nBEST\n'+str(best)
    plt.text(0.01,0.99,txt,va='top',family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

print('Saved',pdf_name)
