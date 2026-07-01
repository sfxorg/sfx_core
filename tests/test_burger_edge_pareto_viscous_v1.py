# test_burger_edge_pareto_viscous_v1.py
# Viscous Burgers + Spectral Edge Pareto Study

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N = 256
L = 1.0
MID = N // 2
DT = 2e-5
NSTEPS = 1000
NU = 1e-3

WIDTHS = [2,4,8,16,32]
RANKS = [1,2,4,8,16]
TOLS = [1e-4,1e-6,1e-8]

x = np.linspace(0,L,N,endpoint=False)
dx = L/N

ik_full = 1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel = 1j*2*np.pi*np.fft.fftfreq(MID,d=dx)
k2_full = -(2*np.pi*np.fft.fftfreq(N,d=dx))**2
k2_panel = -(2*np.pi*np.fft.fftfreq(MID,d=dx))**2

CASES = {
    'Gaussian' : np.exp(-((x-0.25)**2)/(0.03**2)),
    'Sin8' : np.sin(2*np.pi*8*x),
    'Broadband' : np.sin(2*np.pi*8*x)
                  +0.5*np.sin(2*np.pi*16*x)
                  +0.25*np.sin(2*np.pi*32*x)
}


def burgers_viscous_rhs(u, ik, k2):
    uhat = np.fft.fft(u)
    ux = np.fft.ifft(ik*uhat).real
    uxx = np.fft.ifft(k2*uhat).real
    return -(u*ux) + NU*uxx


def rk4(u, rhs, ik, k2):
    k1 = rhs(u,ik,k2)
    k2v = rhs(u+0.5*DT*k1,ik,k2)
    k3 = rhs(u+0.5*DT*k2v,ik,k2)
    k4 = rhs(u+DT*k3,ik,k2)
    return u + DT*(k1+2*k2v+2*k3+k4)/6.0


def edge_step(uA,uB,W,K):
    spec = 0.5*(np.fft.rfft(uA[-W:]) + np.fft.rfft(uB[:W]))

    energy = np.abs(spec)**2
    cume = np.cumsum(energy)/(energy.sum()+1e-30)

    k95 = np.searchsorted(cume,0.95)+1
    k99 = np.searchsorted(cume,0.99)+1

    trunc = spec.copy()
    trunc[min(K,len(trunc)):] = 0

    edge = np.fft.irfft(trunc,n=W)

    ua = uA.copy()
    ub = uB.copy()
    ua[-W:] = edge
    ub[:W] = edge

    return ua,ub,k95,k99

pdf_name = 'test_burger_edge_pareto_viscous_v1.pdf'
summary = []

with PdfPages(pdf_name) as pdf:

    for cname,u0 in CASES.items():

        print('Running',cname)

        ref = u0.copy()
        for _ in range(NSTEPS):
            ref = rk4(ref, burgers_viscous_rhs, ik_full, k2_full)

        pts = []
        histories = {}

        for W in WIDTHS:
            for K in RANKS:

                uA = u0[:MID].copy()
                uB = u0[MID:].copy()

                k95_hist=[]
                k99_hist=[]

                stable=True

                for _ in range(NSTEPS):
                    uA,uB,k95,k99 = edge_step(uA,uB,W,K)

                    uA = rk4(uA, burgers_viscous_rhs, ik_panel, k2_panel)
                    uB = rk4(uB, burgers_viscous_rhs, ik_panel, k2_panel)

                    if not (np.all(np.isfinite(uA)) and np.all(np.isfinite(uB))):
                        stable=False
                        break

                    k95_hist.append(k95)
                    k99_hist.append(k99)

                if not stable:
                    err=np.inf
                    avg95=np.nan
                    avg99=np.nan
                else:
                    sol=np.concatenate([uA,uB])
                    err=np.max(np.abs(sol-ref))
                    avg95=np.mean(k95_hist)
                    avg99=np.mean(k99_hist)

                cost=W*K
                pts.append((W,K,cost,err,avg95,avg99))
                histories[(W,K)] = (k95_hist,k99_hist)

        pareto=[]
        for p in pts:
            if not np.isfinite(p[3]):
                continue
            dominated=False
            for q in pts:
                if not np.isfinite(q[3]):
                    continue
                if (q[2]<=p[2] and q[3]<=p[3]) and (q[2]<p[2] or q[3]<p[3]):
                    dominated=True
                    break
            if not dominated:
                pareto.append(p)

        fig,ax=plt.subplots(1,2,figsize=(11,4))

        valid=[p for p in pts if np.isfinite(p[3])]

        ax[0].loglog([p[2] for p in valid],[p[3] for p in valid],'o',alpha=0.5,label='All')
        ax[0].loglog([p[2] for p in pareto],[p[3] for p in pareto],'ro',label='Pareto')
        ax[0].legend()
        ax[0].set_title('Error vs Cost')
        ax[0].set_xlabel('Cost=W*K')
        ax[0].set_ylabel('Max Error')

        candidates=[p for p in valid if p[3] < 1e-6]
        if candidates:
            best=min(candidates,key=lambda p:p[2])
            k95h,k99h=histories[(best[0],best[1])]
            ax[1].plot(k95h,label='k95')
            ax[1].plot(k99h,label='k99')
            ax[1].set_title(f'Best W={best[0]} K={best[1]}')
            ax[1].legend()

        fig.suptitle(f'{cname}  (nu={NU})')
        pdf.savefig(fig)
        plt.close(fig)

        for tol in TOLS:
            cand=[p for p in valid if p[3] < tol]
            if cand:
                b=min(cand,key=lambda p:p[2])
                summary.append((cname,tol,b[0],b[1],b[2],b[3],b[4],b[5]))

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='VISCOUS BURGERS EDGE PARETO V1\n\n'
    txt+='Case Tol W K Cost Error avgk95 avgk99\n\n'

    for s in summary:
        txt += f'{s[0]:10s} {s[1]:.0e} {s[2]:2d} {s[3]:2d} {s[4]:4d} {s[5]:.3e} {s[6]:.2f} {s[7]:.2f}\n'

    plt.text(0.01,0.99,txt,va='top',family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

print('Saved', pdf_name)
