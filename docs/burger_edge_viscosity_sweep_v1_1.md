# Burgers Edge Viscosity Sweep V1.1

## Date

July 2026

---

# Objective

Investigate whether interface complexity depends on physical dissipation.

The central hypothesis is:

```text
Lower viscosity

↓

More nonlinear spectral generation

↓

Greater interface complexity
```

The study seeks to connect:

```text
PDE physics
```

with

```text
Spectral Edge complexity metrics.
```

---

# Governing Equation

Viscous Burgers equation:

```math
u_t + u u_x = \nu u_{xx}
```

---

# Domain Configuration

## Grid

```text
N = 256
```

---

## Interface Representation

The domain is decomposed into:

```text
Panel A
|
Spectral Edge
|
Panel B
```

using a two-panel Spectral Edge interface.

---

# Initial Condition

Broadband signal:

```text
sin(8x)
+
0.5 sin(16x)
+
0.25 sin(32x)
```

This signal was selected because it contains multiple active frequencies and therefore provides a strong test of interface complexity.

---

# Viscosity Sweep

The following viscosities were tested:

```text
ν = 1e-1
ν = 1e-2
ν = 1e-3
```

---

# Width Sweep

```text
2
4
8
16
32
```

---

# Rank Sweep

```text
1
2
4
8
16
```

---

# Selection Rule

Interfaces were selected using:

```text
Minimum Cost

subject to

Error < 1e-6
```

where:

```text
Cost = Width × Rank
```

---

# Interface Complexity Metrics

## k95

Number of spectral modes required to capture:

```text
95%
```

of interface energy.

---

## k99

Number of spectral modes required to capture:

```text
99%
```

of interface energy.

---

# Stability Results

Every tested configuration remained stable.

Results:

```text
ν = 1e-1
25 / 25 stable
```

```text
ν = 1e-2
25 / 25 stable
```

```text
ν = 1e-3
25 / 25 stable
```

The stabilized viscosity sweep successfully eliminated the numerical failures observed in earlier experiments. 【1-663aeb】

---

# Results

## ν = 1e-1

### Optimal Interface

```text
Width = 32

Rank = 16
```

### Cost

```text
512
```

### Error

```text
5.41e-15
```

### Complexity

```text
k95 = 2.09

k99 = 2.23
```【1-663aeb】

---

## ν = 1e-2

### Optimal Interface

```text
Width = 32

Rank = 16
```

### Cost

```text
512
```

### Error

```text
5.02e-14
```

### Complexity

```text
k95 = 2.52

k99 = 3.29
```【1-663aeb】

---

## ν = 1e-3

### Optimal Interface

```text
Width = 32

Rank = 16
```

### Cost

```text
512
```

### Error

```text
7.37e-14
```

### Complexity

```text
k95 = 3.00

k99 = 5.44
```

【1-663aeb】

---

# Summary Table

| Viscosity | Width | Rank | Cost | Error | k95 | k99 |
|------------|-------:|-------:|-------:|-------:|-------:|-------:|
| 1e-1 | 32 | 16 | 512 | 5.41e-15 | 2.09 | 2.23 |
| 1e-2 | 32 | 16 | 512 | 5.02e-14 | 2.52 | 3.29 |
| 1e-3 | 32 | 16 | 512 | 7.37e-14 | 3.00 | 5.44 |

Results obtained from the viscosity sweep study. 【1-663aeb】

---

# Major Finding

## Interface Complexity Depends on Viscosity

The primary observation is:

```text
As viscosity decreases,

k95 increases

k99 increases.
```

Observed trend:

```text
k95

2.09
→
2.52
→
3.00
```

and:

```text
k99

2.23
→
3.29
→
5.44
```

for:

```text
ν = 1e-1
ν = 1e-2
ν = 1e-3
```

respectively. 【1-663aeb】

---

# Physical Interpretation

Lower viscosity provides weaker dissipation.

Consequently:

```text
More high-frequency content survives.

↓

More spectral energy reaches the interface.

↓

Interface complexity increases.
```

The measured increase in:

```text
k95
```

and

```text
k99
```

is consistent with this interpretation. 【1-663aeb】

---

# Secondary Observation

The optimal interface:

```text
Width = 32

Rank = 16
```

remained unchanged across all tested viscosities. 【1-663aeb】

This suggests that:

```text
Current search-space resolution
may be insufficient to detect
viscosity-dependent optimal cost.
```

However:

```text
Complexity metrics
clearly changed.
```

---

# Comparison With Previous Studies

## Linear Advection

Previous studies demonstrated:

```text
Highly compressed interfaces
with very small ranks.
```

---

## Inviscid Burgers

Earlier inviscid experiments exhibited:

```text
Interface complexity growth

and

numerical instability.
```

---

## Viscous Burgers

The present study demonstrates:

```text
Stable nonlinear evolution

with measurable
complexity growth.
```【1-e50c2f】【1-663aeb】

---

# Scientific Significance

This is the first experiment to establish a direct connection between:

```text
PDE dissipation
```

and

```text
Spectral Edge complexity.
```

The results suggest that interface complexity is not purely geometric.

Instead:

```text
Interface complexity depends on
the underlying physical dynamics.
```

【1-663aeb】

---

# Working Hypothesis

A preliminary complexity relation may exist:

```text
Lower Viscosity

↓

Higher Spectral Complexity

↓

Larger Interface Complexity
```

where complexity is measured using:

```text
k95
```

and

```text
k99.
```

---

# Current Status

| Item | Status |
|--------|--------|
| Linear Spectral Edge | ✅ |
| Pareto Interface Selection | ✅ |
| Inviscid Burgers Exploration | ✅ |
| Viscous Burgers Validation | ✅ |
| Viscosity Sweep | ✅ |
| Physics-Complexity Coupling | ✅ |

---

# Future Work

## Extended Viscosity Range

Investigate:

```text
ν < 1e-3
```

using additional stabilization strategies.

---

## Adaptive Rank Selection

Determine whether:

```text
Rank
```

can be predicted directly from:

```text
k95
```

or

```text
k99.
```

---

## Wave Crossing Experiments

Initialize waves entirely within a single panel and measure:

```text
Transmission across the interface.
```

---

## Multi-Interface Systems

Extend from:

```text
2 panels
```

to:

```text
4 panels

8 panels

many interfaces
```

and study complexity growth.

---

# Conclusion

The viscosity sweep provides the strongest evidence so far that Spectral Edge complexity is influenced by the underlying physics.

While the optimal interface parameters remained unchanged over the tested viscosity range, the interface complexity metrics increased systematically as viscosity decreased:

```text
Lower viscosity

↓

Higher k95

↓

Higher k99
```

This result supports the view that Spectral Edge complexity is connected to spectral-content generation within the governing PDE and represents an important step toward a physics-based theory of compressed spectral interfaces. 【1-663aeb】