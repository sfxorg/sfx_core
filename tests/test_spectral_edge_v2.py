# test_spectral_edge_v2.py
# Consolidated Spectral Edge Study
# V5.1-V5.5 combined

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

N = 256
L = 1.0
MID = N // 2
DT = 2e-4
NSTEPS = 2000
C = 1.0

WIDTHS = [2,4,8,16,32]
RANKS = [1,2,4,8,16]
ADAPT_TOLS = [0.95,0.99,0.999]

x = np.linspace(0.0,L,N,endpoint=False)
dx = L/N
ik = 1j*2*np.pi*np.fft.fftfreq(N,d=dx)


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

CASES = {
    'Gaussian': gaussian(),
    'Sin8': sinusoid(8),
    'Sin32': sinusoid(32),
    'Broadband': broadband(),
}


def fft_rhs(u):
    uhat = np.fft.fft(u)
    ux = np.fft.ifft(ik*uhat).real
    return -C*ux


def rk4(rhs,u):
    k1=rhs(u)
    k2=rhs(u+0.5*DT*k1)
    k3=rhs(u+0.5*DT*k2)
    k4=rhs(u+DT*k3)
    return u + DT*(k1+2*k2+2*k3+k4)/6.0


def edge_rank_mode(uA,uB,W,K=None,adaptive=None):
    left = uA[-W:].copy()
    right = uB[:W].copy()

    LA = np.fft.rfft(left)
    RB = np.fft.rfft(right)
    avg = 0.5*(LA+RB)

    energy = np.abs(avg)**2

    if adaptive is not None:
        cum = np.cumsum(energy)/(energy.sum()+1e-30)
        K = np.searchsorted(cum,adaptive)+1

    if K is not None:
        K = min(K,len(avg))
        avg[K:] = 0

    rec = np.fft.irfft(avg,n=W)

    uA[-W:] = rec
    uB[:W] = rec

    cum = np.cumsum(energy)/(energy.sum()+1e-30)
    k95 = np.searchsorted(cum,0.95)+1
    k99 = np.searchsorted(cum,0.99)+1

    return uA,uB,k95,k99,energy.sum(),K


pdf_name = 'test_spectral_edge_v2.pdf'
summary=[]

with PdfPages(pdf_name) as pdf:

    fig = plt.figure(figsize=(11,8))
    plt.axis('off')
    plt.text(0.05,0.9,'Spectral Edge Study V2',fontsize=18)
    plt.text(0.05,0.75,f'Cases: {list(CASES.keys())}')
    plt.text(0.05,0.65,f'Widths: {WIDTHS}')
    plt.text(0.05,0.55,f'Ranks: {RANKS}')
    plt.text(0.05,0.45,f'Adaptive Tolerances: {ADAPT_TOLS}')
    pdf.savefig(fig)
    plt.close(fig)

    for cname,u0 in CASES.items():

        u_ref = u0.copy()
        for _ in range(NSTEPS):
            u_ref = rk4(fft_rhs,u_ref)

        records=[]

        for W in WIDTHS:

            for K in RANKS:

                uA = u0[:MID].copy()
                uB = u0[MID:].copy()

                k95_hist=[]
                k99_hist=[]
                e_hist=[]

                for _ in range(NSTEPS):
                    uA,uB,k95,k99,e,_ = edge_rank_mode(uA,uB,W,K=K)
                    uA = rk4(lambda q: fft_rhs(np.pad(q,(0,MID)))[:MID],uA)
                    #uB = rk4(lambda q: fft_rhs(np.pad(q,(MID,0))[MID:],uB)

                    uB = rk4(
                        lambda q: fft_rhs(
                            np.pad(q, (MID, 0))[MID:]
                        ),
                        uB
                    )
                    
                    k95_hist.append(k95)
                    k99_hist.append(k99)
                    e_hist.append(e)

                u_two = np.concatenate([uA,uB])

                maxerr = np.max(np.abs(u_ref-u_two))
                l2err = np.sqrt(np.mean((u_ref-u_two)**2))
                masserr = abs(np.sum(u_ref)-np.sum(u_two))*dx

                records.append((W,K,maxerr,l2err,masserr,
                                np.mean(k95_hist),np.mean(k99_hist)))

            for tol in ADAPT_TOLS:

                uA = u0[:MID].copy()
                uB = u0[MID:].copy()
                ranks=[]

                for _ in range(250):
                    uA,uB,_,_,_,r = edge_rank_mode(
                        uA,uB,W,adaptive=tol
                    )
                    ranks.append(r)
                    uA = rk4(lambda q: fft_rhs(np.pad(q,(0,MID)))[:MID],uA)
                    #uB = rk4(lambda q: fft_rhs(np.pad(q,(MID,0))[MID:],uB)

                    uB = rk4(
                        lambda q: fft_rhs(
                            np.pad(q, (MID, 0))[MID:]
                        ),
                        uB
                    )

                summary.append((cname,W,tol,np.mean(ranks)))

        fig,axs = plt.subplots(2,2,figsize=(10,8))

        ws=[r[0] for r in records]
        ks=[r[1] for r in records]
        errs=[r[2] for r in records]
        k95s=[r[5] for r in records]

        axs[0,0].scatter(ws,errs,c=ks)
        axs[0,0].set_title('Max Error')

        axs[0,1].scatter(ws,k95s,c=ks)
        axs[0,1].set_title('Average k95')

        axs[1,0].plot(range(len(errs)),errs)
        axs[1,0].set_yscale('log')
        axs[1,0].set_title('Error Sweep')

        axs[1,1].plot(range(len(k95s)),k95s)
        axs[1,1].set_title('Compressibility Sweep')

        fig.suptitle(cname)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    fig = plt.figure(figsize=(11,8))
    plt.axis('off')
    txt='Adaptive Rank Summary\n\n'
    for row in summary[:50]:
        txt += f'{row}\n'
    plt.text(0.01,0.99,txt,va='top',family='monospace')
    pdf.savefig(fig)
    plt.close(fig)

print('Saved',pdf_name)
