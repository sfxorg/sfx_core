# 05_SFX_Flux_Schwarz_Transmission_V7.py
# Flux-Based Schwarz Transmission

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NX=128
NY=128
MID=NX//2
NT=250
N_SCHWARZ=5
CFL=0.35
VX=1.0
VY=0.5
SIGMA=0.05
TOLS=[1e-2,1e-3,1e-4]
PDF='05_SFX_Flux_Schwarz_Transmission_V7.pdf'

x=np.linspace(0,1,NX,endpoint=False)
y=np.linspace(0,1,NY,endpoint=False)
X,Y=np.meshgrid(x,y,indexing='ij')

u0=np.exp(-((X-0.25)**2+(Y-0.5)**2)/SIGMA**2)

def full_step(q):
    qx=q-np.roll(q,1,axis=0)
    qy=q-np.roll(q,1,axis=1)
    return q-CFL*VX*qx-CFL*VY*qy

def split(q):
    return q[:MID,:].copy(), q[MID:,:].copy()

def merge(a,b):
    return np.vstack([a,b])

def physical_flux(trace):
    return 0.5*trace*trace

def rusanov_flux(left_trace,right_trace):
    lam=np.maximum(np.abs(left_trace),np.abs(right_trace))
    fL=physical_flux(left_trace)
    fR=physical_flux(right_trace)
    return 0.5*(fL+fR)-0.5*lam*(right_trace-left_trace)

def fft_compress(v,tol):
    c=np.fft.rfft(v)
    e=np.abs(c)**2
    tot=e.sum()+1e-16
    run=0.0
    rank=len(c)
    for k in range(1,len(c)+1):
        run+=e[k-1]
        if np.sqrt((tot-run)/tot)<=tol:
            rank=k
            break
    cc=c.copy(); cc[rank:]=0
    return np.fft.irfft(cc,n=len(v)), rank

def schwarz_flux_step(left,right,tol):
    ranks=[]
    residuals=[]
    for _ in range(N_SCHWARZ):
        ltr=left[-1,:]
        rtr=right[0,:]
        flux=rusanov_flux(ltr,rtr)
        flux_rec,rank=fft_compress(flux,tol)
        corr=0.10*flux_rec
        left[-1,:]+=corr
        right[0,:]-=corr
        residuals.append(np.linalg.norm(ltr-rtr))
        ranks.append(rank)
    q=full_step(merge(left,right))
    left,right=split(q)
    return left,right,max(ranks),residuals[-1]

ref=u0.copy(); refhist=[]
for _ in range(NT):
    ref=full_step(ref)
    refhist.append(ref.copy())

results=[]
with PdfPages(PDF) as pdf:
    for tol in TOLS:
        left,right=split(u0.copy())
        errs=[]; ranks=[]; res=[]
        for n in range(NT):
            left,right,r,resid=schwarz_flux_step(left,right,tol)
            q=merge(left,right)
            err=np.linalg.norm(q-refhist[n])/(np.linalg.norm(refhist[n])+1e-16)
            errs.append(err); ranks.append(r); res.append(resid)
        results.append((tol,errs[-1],np.mean(ranks),res[-1]))
        fig,ax=plt.subplots(1,3,figsize=(12,4))
        ax[0].semilogy(np.maximum(errs,1e-16)); ax[0].set_title('L2 Error')
        ax[1].plot(ranks); ax[1].set_title('FFT Rank')
        ax[2].semilogy(np.maximum(res,1e-16)); ax[2].set_title('Interface Residual')
        fig.suptitle(f'Flux Schwarz tol={tol:.0e}')
        pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=(11,8))
    plt.axis('off')

    summary_lines = [
        "SFX V7 FLUX SCHWARZ TRANSMISSION",
        "",
        "Tol FinalErr AvgRank FinalResidual",
        ""
    ]

    for t, e, r, res in results:
        summary_lines.append(
            f"{t:.0e} {e:.6e} {r:.2f} {res:.6e}"
        )

    txt = "\n".join(summary_lines)

    plt.text(
        0.02,
        0.98,
        txt,
        va='top',
        family='monospace'
    )

    pdf.savefig(fig)
plt.close(fig)

print("Saved", PDF)
