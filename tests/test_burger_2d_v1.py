# test_burger_2d_v1.py
# Extended diagnostics version

import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.jacobians import sem_jacobian
from operators.burgers_ops import (
    burgers_fft_rhs,
    burgers_fv_rhs_3rd_order,
    burgers_hybrid_standard,
)

# ============================================================
# USER PARAMETERS
# ============================================================
L = 10.0
N = 1024
P = 8

dx = L / N
dt = 1.0e-4
total_time = 0.5
steps = int(total_time / dt)
SAMPLE_EVERY = 50

# ============================================================
# HELPERS
# ============================================================

def rk4_step(rhs, u, dt):
    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def spectral_rank(signal):
    spec = jnp.fft.rfft(signal)
    e = jnp.abs(spec)**2
    e = e / (jnp.sum(e) + 1e-30)
    cume = jnp.cumsum(e)
    k95 = int(jnp.argmax(cume > 0.95))
    k99 = int(jnp.argmax(cume > 0.99))
    return k95, k99

# ============================================================
# GRID
# ============================================================
X, Y = jnp.meshgrid(
    jnp.linspace(0.0, L, N, endpoint=False),
    jnp.linspace(0.0, L, N, endpoint=False),
    indexing='ij'
)

ik_x = (1j * 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=L/N))[:,None]
ik_y = (1j * 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=L/N))[None,:]

u_init = jnp.exp(-((X-5.0)**2 + (Y-5.0)**2)/1.5)

_, D_rib = get_sem_diff_matrix_2d(P)
jac_rib = sem_jacobian(16, L)

# ============================================================
# FFT REFERENCE
# ============================================================
@jax.jit
def step_fft(u,_):
    return rk4_step(lambda q: burgers_fft_rhs(q,ik_x,ik_y),u,dt), None

print('\nRunning FFT reference ...')
t0=time.perf_counter()
u_fft = jax.lax.scan(step_fft,jnp.copy(u_init),None,length=steps)[0]
jax.block_until_ready(u_fft)
t_fft=time.perf_counter()-t0

# ============================================================
# HYBRID WITH TIME-HISTORY DIAGNOSTICS
# ============================================================
state=(
    jnp.copy(u_init),
    1.1*u_init[:P+1,:],
    1.1*u_init[-P-1:,:],
    1.1*u_init[:,:P+1],
    1.1*u_init[:,-P-1:]
)

# warmup
state = burgers_hybrid_standard(
    state[0],state[1],state[2],state[3],state[4],
    ik_x,ik_y,D_rib,jac_rib,dt,dx
)
jax.block_until_ready(state[0])

# restart
state=(
    jnp.copy(u_init),
    1.1*u_init[:P+1,:],
    1.1*u_init[-P-1:,:],
    1.1*u_init[:,:P+1],
    1.1*u_init[:,-P-1:]
)

left_k95_hist=[]; left_k99_hist=[]
right_k95_hist=[]; right_k99_hist=[]
top_k95_hist=[]; top_k99_hist=[]
bot_k95_hist=[]; bot_k99_hist=[]
rib_energy_hist=[]

print('Running HYBRID diagnostics ...')
t0=time.perf_counter()
for n in range(steps):

    state = burgers_hybrid_standard(
        state[0],state[1],state[2],state[3],state[4],
        ik_x,ik_y,D_rib,jac_rib,dt,dx
    )

    if n % SAMPLE_EVERY == 0:
        u=state[0]

        top_err = state[1][0:2,:] - u[-2:,:]
        bot_err = state[2][-2:,:] - u[0:2,:]
        left_err = state[3][:,-2:] - u[:,0:2]
        right_err = state[4][:,0:2] - u[:,-2:]

        k95,k99=spectral_rank(jnp.mean(top_err,axis=0))
        top_k95_hist.append(k95); top_k99_hist.append(k99)

        k95,k99=spectral_rank(jnp.mean(bot_err,axis=0))
        bot_k95_hist.append(k95); bot_k99_hist.append(k99)

        k95,k99=spectral_rank(jnp.mean(left_err.T,axis=0))
        left_k95_hist.append(k95); left_k99_hist.append(k99)

        k95,k99=spectral_rank(jnp.mean(right_err.T,axis=0))
        right_k95_hist.append(k95); right_k99_hist.append(k99)

        rib_energy=(
            jnp.mean(state[1]**2)+jnp.mean(state[2]**2)+
            jnp.mean(state[3]**2)+jnp.mean(state[4]**2)
        )
        rib_energy_hist.append(float(rib_energy))
    

jax.block_until_ready(state[0])
t_hybrid=time.perf_counter()-t0

u_hybrid=state[0]

# ============================================================
# SUMMARY
# ============================================================
err=jnp.abs(u_hybrid-u_fft)

print('\n' + '='*60)
print('HYBRID SUMMARY')
print('='*60)
print('FFT runtime    :',t_fft)
print('HYB runtime    :',t_hybrid)
print('Max error      :',float(jnp.max(err)))
print('L2 error       :',float(jnp.sqrt(jnp.mean(err**2))))
print('Mass diff      :',float(abs(jnp.mean(u_hybrid)-jnp.mean(u_fft))))
print('Energy diff    :',float(abs(jnp.mean(u_hybrid**2)-jnp.mean(u_fft**2))))

for name,k95,k99 in [
    ('LEFT',left_k95_hist,left_k99_hist),
    ('RIGHT',right_k95_hist,right_k99_hist),
    ('TOP',top_k95_hist,top_k99_hist),
    ('BOTTOM',bot_k95_hist,bot_k99_hist),
]:
    print('\n'+name)
    print('avg k95 =',float(np.mean(k95)))
    print('max k95 =',int(np.max(k95)))
    print('avg k99 =',float(np.mean(k99)))
    print('max k99 =',int(np.max(k99)))

# ============================================================
# PLOTS
# ============================================================
plt.figure(figsize=(10,6))
plt.plot(left_k95_hist,label='Left k95')
plt.plot(right_k95_hist,label='Right k95')
plt.plot(top_k95_hist,label='Top k95')
plt.plot(bot_k95_hist,label='Bottom k95')
plt.legend()
plt.xlabel('Sample')
plt.ylabel('95% spectral rank')
plt.title('Interface Spectral Complexity History')
plt.tight_layout()
plt.savefig('interface_rank_history.png',dpi=200)
plt.close()

plt.figure(figsize=(10,4))
plt.semilogy(np.maximum(rib_energy_hist,1e-30))
plt.xlabel('Sample')
plt.ylabel('Ribbon energy')
plt.title('Ribbon Energy History')
plt.tight_layout()
plt.savefig('ribbon_energy_history.png',dpi=200)
plt.close()

print('\nSaved: interface_rank_history.png')
print('Saved: ribbon_energy_history.png')
