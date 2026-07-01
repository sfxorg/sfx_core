
# 03_Adaptive_Compression_Suite_v5.py
# V5: Singular-value based POD rank selection.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

NX=128
NY=128
MID=NX//2
NT=250
CFL=0.35
VX=1.0
VY=0.5
SIGMA=0.05
TOLS=[1e-1,1e-2,1e-3,1e-4,1e-5]
TRAIN_STEPS=1000

x=np.linspace(0,1,NX,endpoint=False)
y=np.linspace(0,1,NY,endpoint=False)
X,Y=np.meshgrid(x,y,indexing='ij')

# Cases

def gaussian():
    return np.exp(-((X-0.25)**2+(Y-0.50)**2)/SIGMA**2)

def sinusoid(k):
    return gaussian()*np.sin(2*np.pi*k*X)

def broadband():
    env=gaussian()
    return env*(np.sin(2*np.pi*8*X)+0.7*np.sin(2*np.pi*16*X)+0.5*np.sin(2*np.pi*32*X)+0.3*np.sin(2*np.pi*64*X))

def random_packet():
    rng=np.random.default_rng(1)
    q=np.zeros((NX,NY))
    for k in [4,8,16,24,32,48]:
        q += rng.uniform(0.2,1.0)*np.sin(2*np.pi*k*X+rng.uniform(0,2*np.pi))
    return q*gaussian()

CASES={
'Gaussian':gaussian(),
'Sin8':sinusoid(8),
'Sin32':sinusoid(32),
'Broadband':broadband(),
'Random':random_packet()
}

TRAIN_CASES={
'Gaussian':gaussian(),
'Sin8':sinusoid(8),
'Sin32':sinusoid(32),
'Sin64':sinusoid(64),
'Sin96':sinusoid(96),
'Broadband':broadband(),
'Random':random_packet()
}

# Solver

def full_step(q):
    qx=q-np.roll(q,1,axis=0)
    qy=q-np.roll(q,1,axis=1)
    return q - CFL*VX*qx - CFL*VY*qy

def split(q):
    return q[:MID,:].copy(), q[MID:,:].copy()

def merge(l,r):
    return np.vstack([l,r])

# POD training
snaps=[]
for _,u0 in TRAIN_CASES.items():
    q=u0.copy()
    for _ in range(TRAIN_STEPS):
        q=full_step(q)
        snaps.append(q[MID-1,:].copy())
        snaps.append(q[-1,:].copy())

S=np.asarray(snaps)
POD_MEAN=np.mean(S,axis=0)
A=(S-POD_MEAN).T
U,Sigma,VT=np.linalg.svd(A,full_matrices=False)
POD_ENERGY=np.cumsum(Sigma**2)
POD_ENERGY/=POD_ENERGY[-1]

# tolerance -> rank from singular values

def pod_rank_from_tol(tol):
    target=1.0-tol**2
    return max(1,np.searchsorted(POD_ENERGY,target)+1)

# FFT

def fft_adaptive(trace,tol):
    c=np.fft.rfft(trace)
    e=np.abs(c)**2
    tot=e.sum(); run=0.0
    for r in range(1,len(c)+1):
        run+=e[r-1]
        if np.sqrt((tot-run)/(tot+1e-16))<=tol:
            cc=c.copy(); cc[r:]=0
            return np.fft.irfft(cc,n=len(trace)),r
    return trace.copy(),len(c)

# POD V5 singular-value rank selection

def pod_adaptive(trace,tol):
    r=pod_rank_from_tol(tol)
    centered=trace-POD_MEAN
    rec=U[:,:r]@(U[:,:r].T@centered)+POD_MEAN
    return rec,r


def transfer(trace,method,tol):
    return fft_adaptive(trace,tol) if method=='FFT' else pod_adaptive(trace,tol)


def step(left,right,method,tol):
    gr,r1=transfer(left[-1,:],method,tol)
    gl,r2=transfer(right[-1,:],method,tol)
    q=merge(left,right)
    q[MID-1,:]=gr
    q[-1,:]=gl
    q=full_step(q)
    l,r=split(q)
    return l,r,max(r1,r2)

pdf='03_Adaptive_Compression_Suite_v5.pdf'
with PdfPages(pdf) as out:
    for cname,u0 in CASES.items():
        ref=u0.copy(); hist=[]
        for _ in range(NT):
            ref=full_step(ref); hist.append(ref.copy())
        for method in ['FFT','POD']:
            for tol in TOLS:
                left,right=split(u0.copy())
                errs=[]; ranks=[]
                for n in range(NT):
                    left,right,r=step(left,right,method,tol)
                    ranks.append(r)
                    q=merge(left,right)
                    errs.append(np.linalg.norm(q-hist[n])/(np.linalg.norm(hist[n])+1e-16))
                fig,ax=plt.subplots(1,2,figsize=(10,4))
                ax[0].semilogy(np.maximum(errs,1e-16))
                ax[1].plot(ranks)
                ax[0].set_title('L2 Error')
                ax[1].set_title('Rank History')
                fig.suptitle(f'{cname} | {method} | tol={tol:.0e} | err={errs[-1]:.3e}')
                fig.tight_layout()
                out.savefig(fig)
                plt.close(fig)
print('Saved',pdf)
