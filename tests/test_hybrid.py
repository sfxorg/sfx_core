import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

# Ensure sfx_core is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local sfx_core modules
from geometry.sem_nodes import get_sem_diff_matrix_2d
from geometry.sem_grids import generate_tiled_sem_grid_2d
from geometry.jacobians import sem_jacobian
from operators.hybrid_ops import run_hybrid_sfx_2d_stable

# Precision configuration: Essential for spectral stability
jax.config.update("jax_enable_x64", True)

def run_forecast():
    # --- 1. CONFIGURATION (Matched to test_master_collision.py) ---
    L, cx, cy, dt = 10.0, 1.0, 1.0, 0.01
    E_sem, P_sem = 32, 4
    num_steps = 500
    
    # Grid size MUST be 160 to match the state shape observed in your error logs
    N_fft = 160 
    
    # --- 2. FFT FREQUENCY VECTORS ---
    # These must match the grid size (160) to avoid broadcasting errors
    ik_x = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[:, None]
    ik_y = 1j * 2 * jnp.pi * jnp.fft.fftfreq(N_fft, d=L/N_fft)[None, :]

    # --- 3. GEOMETRY SETUP ---
    print("Initializing SEM geometry (Matching reference)...")
    X_sem, Y_sem = generate_tiled_sem_grid_2d(E_sem, P_sem, L)
    
    # Use ribbons matching your reference
    P_rib = 4
    _, D_rib = get_sem_diff_matrix_2d(P_rib)
    jac_rib = sem_jacobian(16, L)

    # --- 4. INITIAL STATE ---
    u_init = jnp.exp(-((X_sem - 5)**2 + (Y_sem - 5)**2) / 1.5)
    
    # State tuple: [r, top, bot, left, right]
    current_state = (
        jnp.copy(u_init),
        u_init[:P_rib+1, :],
        u_init[-P_rib-1:, :],
        u_init[:, :P_rib+1],
        u_init[:, -P_rib-1:]
    )

    history = []
    print(f"Generating training data for {num_steps} steps...")

    # --- 5. DATA GENERATION LOOP ---
    try:
        for step in range(num_steps):
            # Hybrid stable step
            next_state = run_hybrid_sfx_2d_stable(
                current_state[0], current_state[1], current_state[2], 
                current_state[3], current_state[4], 
                ik_x, ik_y, D_rib, jac_rib, cx, cy, dt
            )
            
            # --- STABILITY SENTINEL ---
            r_vector = np.array(next_state[0])
            # If max value > 50, physics is diverging (unstable)
            if np.max(np.abs(r_vector)) > 50.0:
                print(f"CRITICAL: Simulation diverged (unstable) at step {step}.")
                print("Solution: Reduce 'dt' (e.g., dt=0.01) in configuration.")
                sys.exit(1)
            
            history.append(r_vector)
            current_state = next_state
            
            if step % 10 == 0:
                print(f"Completed step {step}")

    except Exception as e:
        print("\n--- CRASH DETECTED ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 6. DUMP DATA ---
    data = np.array(history)
    np.save("training_data.npy", data)
    print(f"\nSuccess! Saved stable data shape: {data.shape}")
    print("You are ready to train.")

if __name__ == "__main__":
    run_forecast()
