# test_burger_edge_viscosity_sweep_v1_1.py
# Robust viscosity sweep with stability reporting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N=256
L=1.0
MID=N//2
DT=1e-5
NSTEPS=1000

VISCOSITIES=[1e-1,1e-2,1e-3]
WIDTHS=[2,4,8,16,32]
RANKS=[1,2,4,8,16]

x=np.linspace(0,L,N,endpoint=False)
dx=L/N

ik_full=1j*2*np.pi*np.fft.fftfreq(N,d=dx)
ik_panel=1j*2*np.pi*np.fft.fftfreq(MID,d=dx)
k2_full=-(2*np.pi*np.fft.fftfreq(N,d=dx))**2
k2_panel=-(2*np.pi*np.fft.fftfreq(MID,d=dx))**2

u0=(np.sin(2*np.pi*8*x)
    +0.5*np.sin(2*np.pi*16*x)
    +0.25*np.sin(2*np.pi*32*x))

pdf_name='test_burger_edge_viscosity_sweep_v1_1.pdf'
summary=[]
trend=[]

with PdfPages(pdf_name) as pdf:

    for NU in VISCOSITIES:

        def rhs(u,ik,k2):
            uhat=np.fft.fft(u)
            ux=np.fft.ifft(ik*uhat).real
            uxx=np.fft.ifft(k2*uhat).real
            return -(u*ux) + NU*uxx

        def rk4(u,ik,k2):
            k1=rhs(u,ik,k2)
            k2v=rhs(u+0.5*DT*k1,ik,k2)
            k3=rhs(u+0.5*DT*k2v,ik,k2)
            k4=rhs(u+DT*k3,ik,k2)
            return u + DT*(k1+2*k2v+2*k3+k4)/6.0

        def edge(uA,uB,W,K):
            spec=0.5*(np.fft.rfft(uA[-W:])+np.fft.rfft(uB[:W]))
            e=np.abs(spec)**2
            c=np.cumsum(e)/(e.sum()+1e-30)
            k95=np.searchsorted(c,0.95)+1
            k99=np.searchsorted(c,0.99)+1
            trunc=spec.copy()
            trunc[min(K,len(trunc)):]=0
            iface=np.fft.irfft(trunc,n=W)
            ua=uA.copy(); ub=uB.copy()
            ua[-W:]=iface
            ub[:W]=iface
            return ua,ub,k95,k99

        print('Running nu=',NU)

        ref=u0.copy()
        stable_ref=True
        for _ in range(NSTEPS):
            ref=rk4(ref,ik_full,k2_full)
            if not np.all(np.isfinite(ref)):
                stable_ref=False
                break

        if not stable_ref:
            summary.append((NU,'REF_FAILED'))
            continue

        pts=[]
        histories={}
        stable_count=0
        total_count=0

        for W in WIDTHS:
            for K in RANKS:
                total_count += 1

                uA=u0[:MID].copy()
                uB=u0[MID:].copy()
                k95h=[]
                k99h=[]

                stable=True

                for _ in range(NSTEPS):
                    uA,uB,k95,k99=edge(uA,uB,W,K)
                    uA=rk4(uA,ik_panel,k2_panel)
                    uB=rk4(uB,ik_panel,k2_panel)

                    if not (np.all(np.isfinite(uA)) and np.all(np.isfinite(uB))):
                        stable=False
                        break

                    k95h.append(k95)
                    k99h.append(k99)

                if not stable:
                    continue

                stable_count += 1
                err=np.max(np.abs(np.concatenate([uA,uB]) - ref))
                cost=W*K
                pts.append((W,K,cost,err,np.mean(k95h),np.mean(k99h)))
                histories[(W,K)]=(k95h,k99h)

        if len(pts)==0:
            summary.append((NU,'NO_STABLE_CONFIG',stable_count,total_count))
            continue

        candidates=[p for p in pts if p[3] < 1e-6]
        best=min(candidates,key=lambda p:p[2]) if candidates else min(pts,key=lambda p:p[3])

        trend.append((NU,best[2],best[4],best[5]))

        fig,ax=plt.subplots(1,2,figsize=(11,4))

        ax[0].loglog([p[2] for p in pts],[p[3] for p in pts],'o')
        ax[0].set_title(f'nu={NU:.0e} stable={stable_count}/{total_count}')
        ax[0].set_xlabel('Cost=W*K')
        ax[0].set_ylabel('Max Error')

        h95,h99=histories[(best[0],best[1])]
        ax[1].plot(h95,label='k95')
        ax[1].plot(h99,label='k99')
        ax[1].legend()
        ax[1].set_title(f'Best W={best[0]} K={best[1]}')

        pdf.savefig(fig)
        plt.close(fig)

        summary.append((NU,best[0],best[1],best[2],best[3],best[4],best[5],stable_count,total_count))

    if len(trend)>0:
        nu=np.array([t[0] for t in trend])
        cost=np.array([t[1] for t in trend])
        k95=np.array([t[2] for t in trend])
        k99=np.array([t[3] for t in trend])

        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].semilogx(nu,cost,'o-')
        ax[0].set_title('Optimal Cost vs Viscosity')

        ax[1].semilogx(nu,k95,'o-',label='k95')
        ax[1].semilogx(nu,k99,'o-',label='k99')
        ax[1].legend()
        ax[1].set_title('Complexity vs Viscosity')
        pdf.savefig(fig)
        plt.close(fig)

    fig=plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='VISCOSITY SWEEP V1.1\n\n'
    for s in summary:
        txt += str(s) + '\n'
    plt.text(0.01,0.99,txt,va='top',family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

print('Saved',pdf_name)
