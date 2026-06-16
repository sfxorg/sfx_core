import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
import sys

# 1. INTEGRITY CHECK (Sanity check before training)
def check_dataset_integrity(name, data):
    print(f"--- Checking {name} set ---")
    if np.isnan(data).any() or np.isinf(data).any():
        print(f"CRITICAL: {name} set contains NaNs or Infs!")
        return False
    print(f"Shape: {data.shape} | Range: [{data.min():.2f}, {data.max():.2f}]")
    return True

# 2. LOAD DATA
try:
    data = np.load("training_data.npy") # Expected: (500, 160, 160)
except FileNotFoundError:
    print("Error: training_data.npy not found.")
    sys.exit(1)

# 3. SPLIT & NORMALIZE
split_idx = int(0.8 * len(data))
train_data, val_data = data[:split_idx], data[split_idx:]

if not check_dataset_integrity("TRAIN", train_data): sys.exit(1)
if not check_dataset_integrity("VAL", val_data): sys.exit(1)

# Normalization using TRAIN stats only
train_mean = np.mean(train_data)
train_std = np.std(train_data) + 1e-8
norm = lambda x: (x - train_mean) / train_std

X_train = norm(train_data[:-1])[..., np.newaxis]
Y_train = norm(train_data[1:])[..., np.newaxis]
X_val = norm(val_data[:-1])[..., np.newaxis]
Y_val = norm(val_data[1:])[..., np.newaxis]

# 4. CNN MODEL
class SurrogateModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=1, kernel_size=(3, 3), padding='SAME')(x)
        return x

# 5. TRAINING SETUP
key = jax.random.PRNGKey(0)
model = SurrogateModel()
params = model.init(key, X_train[:1])

# CLIP BY GLOBAL NORM prevents the NaN errors you saw earlier
tx = optax.chain(
    optax.clip_by_global_norm(0.5), 
    optax.adam(learning_rate=1e-4) 
)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_x, batch_y):
    def loss_fn(params):
        preds = state.apply_fn(params, batch_x)
        return jnp.mean((preds - batch_y) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# 6. TRAINING LOOP
batch_size = 32
print(f"\nStarting training on {len(X_train)} samples...")

for epoch in range(1000):
    # Simple mini-batch shuffling
    perm = jax.random.permutation(key, len(X_train))
    for i in range(0, len(X_train), batch_size):
        batch_idx = perm[i:i+batch_size]
        state, loss = train_step(state, X_train[batch_idx], Y_train[batch_idx])
    
    if epoch % 50 == 0:
        val_preds = state.apply_fn(state.params, X_val)
        val_loss = jnp.mean((val_preds - Y_val) ** 2)
        print(f"Epoch {epoch:4d} | Train Loss: {loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if np.isnan(loss):
            print("NaN detected! Check your learning rate.")
            break
