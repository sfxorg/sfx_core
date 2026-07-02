# SFX Schwarz + Adaptive Transmission V6
# Milestones:
# 1. State Transmission (baseline)
# 2. Flux Transmission
# 3. FFT Compression
# 4. POD Compression
# 5. Schwarz Iteration
# 6. PDF Diagnostics

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NX=128
NY=128
MID=NX//2
NT=250
N_SCHWARZ=3
CFL=0.35
VX=1.0
VY=0.5
SIGMA=0.05
TOLS=[1e-2,1e-3,1e-4]
PDF='04_SFX_Schwarz_Transmission_V6.pdf'

x=np.linspace(0,1,NX,endpoint=False)
y=np.linspace(0,1,NY,endpoint=False)
X,Y=np.meshgrid(x,y,indexing='ij')

u0=np.exp(-((X-0.25)**2+(Y-0.5)**2)/SIGMA**2)

def full_step(q):
    qx=q-np.roll(q,1,axis=0)
    qy=q-np.roll(q,1,axis=1)
    return q-CFL*VX*qx-CFL*VY*qy

def split(q):
    return q[:MID,:].copy(),q[MID:,:].copy()

def merge(a,b):
    return np.vstack([a,b])

def flux_trace(left,right):
    return 0.5*(left[-1,:]+right[0,:])

def fft_compress(v,tol):
    c=np.fft.rfft(v)
    e=np.abs(c)**2
    tot=e.sum()+1e-16
    run=0.0
    r=len(c)
    for k in range(1,len(c)+1):
        run+=e[k-1]
        if np.sqrt((tot-run)/tot)<=tol:
            r=k
            break
    cc=c.copy(); cc[r:]=0
    return np.fft.irfft(cc,n=len(v)),r

def schwarz_step(left,right,tol,use_flux=True):
    rank_hist=[]
    for _ in range(N_SCHWARZ):
        trace=flux_trace(left,right) if use_flux else left[-1,:]
        rec,r=fft_compress(trace,tol)
        rank_hist.append(r)
        left[-1,:]=0.5*(left[-1,:]+rec)
        right[0,:]=0.5*(right[0,:]+rec)
    q=full_step(merge(left,right))
    left,right=split(q)
    return left,right,max(rank_hist)

ref=u0.copy(); refhist=[]
for _ in range(NT):
    ref=full_step(ref); refhist.append(ref.copy())

results=[]
with PdfPages(PDF) as pdf:
    for tol in TOLS:
        left,right=split(u0.copy())
        errs=[]; ranks=[]
        for n in range(NT):
            left,right,r=schwarz_step(left,right,tol,True)
            q=merge(left,right)
            err=np.linalg.norm(q-refhist[n])/(np.linalg.norm(refhist[n])+1e-16)
            errs.append(err); ranks.append(r)
        results.append((tol,errs[-1],np.mean(ranks)))
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].semilogy(np.maximum(errs,1e-16))
        ax[0].set_title('Schwarz Error')
        ax[1].plot(ranks)
        ax[1].set_title('Compression Rank')
        pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=(11,8))
    plt.axis('off')

    txt = (
        'SFX V6 SCHWARZ TRANSMISSION\n\n'
        'Tol FinalErr AvgRank\n\n'
    )

    for r in results:
        txt += f'{r[0]:.0e} {r[1]:.6e} {r[2]:.2f}\n'

    plt.text(
        0.02,
        0.98,
        txt,
        va='top',
        family='monospace'
    )

    pdf.savefig(fig)
plt.close(fig)

print('Saved', PDF)
