# test_spectral_edge_pareto.py
# Pareto Frontier Study for Spectral Edge Interfaces

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N=256
L=1.0
MID=N//2
DT=2e-4
NSTEPS=500
C=1.0

WIDTHS=[2,4,8,16,32,64]
RANKS=[1,2,4,8,16]
TOLS=[1e-4,1e-6,1e-8,1e-10]

x=np.linspace(0,L,N,endpoint=False)
dx=L/N
ik_full=1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel=1j*2*np.pi*np.fft.fftfreq(MID,d=dx)

CASES={
 'Sin8':np.sin(2*np.pi*8*x),
 'Sin32':np.sin(2*np.pi*32*x),
 'Broadband':np.sin(2*np.pi*8*x)+0.5*np.sin(2*np.pi*16*x)+0.25*np.sin(2*np.pi*32*x),
 'Gaussian':np.exp(-((x-0.25)**2)/(0.03**2))
}


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
    trunc=spec.copy()
    trunc[min(K,len(trunc)):]=0
    edge=np.fft.irfft(trunc,n=W)
    uA[-W:]=edge
    uB[:W]=edge
    return uA,uB

pdf='test_spectral_edge_pareto.pdf'

with PdfPages(pdf) as out:
    summary=[]

    for cname,u0 in CASES.items():

        ref=u0.copy()
        for _ in range(NSTEPS):
            ref=rk4(rhs_full,ref)

        pts=[]

        for W in WIDTHS:
            for K in RANKS:

                uA=u0[:MID].copy()
                uB=u0[MID:].copy()

                for _ in range(NSTEPS):
                    uA,uB=spectral_edge(uA,uB,W,K)
                    uA=rk4(rhs_panel,uA)
                    uB=rk4(rhs_panel,uB)

                sol=np.concatenate([uA,uB])
                err=np.max(np.abs(sol-ref))
                cost=W*K
                pts.append((W,K,cost,err))

        pareto=[]
        for p in pts:
            dominated=False
            for q in pts:
                if (q[2] <= p[2] and q[3] <= p[3]) and (q[2] < p[2] or q[3] < p[3]):
                    dominated=True
                    break
            if not dominated:
                pareto.append(p)

        fig,ax=plt.subplots(figsize=(7,5))
        costs=[p[2] for p in pts]
        errs=[p[3] for p in pts]
        ax.loglog(costs,errs,'o',alpha=0.5,label='All')

        pc=[p[2] for p in pareto]
        pe=[p[3] for p in pareto]
        ax.loglog(pc,pe,'ro',label='Pareto')

        for tol in TOLS:
            best=None
            for p in pts:
                if p[3] < tol:
                    if best is None or p[2] < best[2]:
                        best=p

            if best is not None:
                summary.append((cname,tol,best[0],best[1],best[2],best[3]))
                ax.plot(best[2],best[3],'ks',markersize=10)

        ax.set_title(cname)
        ax.set_xlabel('Cost = W*K')
        ax.set_ylabel('Max Error')
        ax.legend()
        out.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='PARETO SUMMARY\n\nCase  Tol      W  K  Cost  Error\n\n'
    for s in summary:
        txt += f'{s[0]:10s} {s[1]:.0e} {s[2]:3d} {s[3]:2d} {s[4]:4d} {s[5]:.3e}\n'
    plt.text(0.01,0.99,txt,va='top',family='monospace')
    out.savefig(fig)
    plt.close(fig)

print("Saved",pdf)
