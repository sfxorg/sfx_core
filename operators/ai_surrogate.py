# Create a new file: operators/ai_surrogate.py
import jax.numpy as jnp

def ai_surrogate_model(u_true, cutoff=0.7):
    """
    Mimics an AI surrogate model (e.g., U-Net/GraphCast).
    It smooths the truth (u_true) to simulate AI-driven blurring.
    """
    # 1. FFT
    u_hat = jnp.fft.fft2(u_true)
    
    # 2. Spectral Cutoff (Damp high frequencies)
    # The AI model 'learns' the large scales but loses the 'edges'
    freqs = jnp.abs(jnp.fft.fftfreq(u_true.shape[0]))
    mask = jnp.exp(-freqs**2 / cutoff**2) # Gaussian smoothing
    
    # 3. Apply mask and IFFT
    u_ai = jnp.fft.ifft2(u_hat * mask).real
    return u_ai
