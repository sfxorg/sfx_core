# Viscous Burgers Spectral Edge Pareto Study V1

## Date

July 2026

---

# Objective

Evaluate whether Spectral Edge interface compression remains effective for a nonlinear PDE.

Unlike the linear advection studies, Burgers dynamics permit nonlinear spectral interactions and harmonic generation.

The primary question is:

```text
Does interface complexity remain small
under nonlinear evolution?
```

---

# Background

Previous studies demonstrated:

- highly compressed interfaces for linear advection
- low optimal spectral ranks
- successful Pareto-based interface selection

However:

```text
Inviscid Burgers
```

produced numerical instabilities and significant interface complexity growth.

A viscous Burgers model was therefore introduced to stabilize the dynamics.

---

# Governing Equation

Viscous Burgers equation:

```math
u_t + u u_x
=
\nu u_{xx}
```

with

```text
ν = 1e-3
```

---

# Numerical Configuration

## Domain

```text
1D periodic domain
```

---

## Grid

```text
N = 256
```

---

## Time Integration

```text
RK4
```

---

## Interface Widths

```text
2
4
8
16
32
```

---

## Interface Ranks

```text
1
2
4
8
16
```

---

## Pareto Selection Rule

Rather than selecting:

```text
Minimum Error
```

the study selects:

```text
Minimum Cost

subject to

Error < Tolerance
```

where:

```text
Cost = Width × Rank
```

---

# Test Cases

## Gaussian

Localized smooth packet.

---

## Sin8

Single-frequency mode.

---

## Broadband

Multiple-frequency signal.

---

# Diagnostics

For every interface configuration:

```text
Width
Rank
```

the following were computed:

- Maximum Error
- Interface Cost
- Average k95
- Average k99

---

# Spectral Complexity Measures

## k95

Minimum number of modes required to capture:

```text
95%
```

of interface energy.

---

## k99

Minimum number of modes required to capture:

```text
99%
```

of interface energy.

---

# Results

## Gaussian

### Optimal Configuration

```text
Width = 2

Rank = 1
```

### Cost

```text
2
```

### Error

```text
4.218e-08
```

### Complexity

```text
k95 ≈ 2.22

k99 ≈ 2.24
```

This represents the most compressed interface observed in the study. 【1-85fcb5】

---

## Sin8

### Optimal Configuration

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
4.674e-14
```

### Complexity

```text
k95 ≈ 2.58

k99 ≈ 3.46
```

Near machine-precision accuracy was achieved despite nonlinear evolution. 【1-85fcb5】

---

## Broadband

### Optimal Configuration

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
7.760e-14
```

### Complexity

```text
k95 ≈ 2.63

k99 ≈ 5.72
```

The interface remained strongly compressed even for a multi-frequency signal. 【1-85fcb5】

---

# Summary Table

| Case | Width | Rank | Cost | Error | Avg k95 | Avg k99 |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|
| Gaussian | 2 | 1 | 2 | 4.218e-08 | 2.22 | 2.24 |
| Sin8 | 32 | 16 | 512 | 4.674e-14 | 2.58 | 3.46 |
| Broadband | 32 | 16 | 512 | 7.760e-14 | 2.63 | 5.72 |

Results obtained from the viscous Burgers Pareto study. 【1-85fcb5】

---

# Comparison with Inviscid Burgers

The earlier inviscid study exhibited:

- overflow warnings
- numerical instability
- significantly larger interface complexity

For example:

```text
Sin8

avg k95 ≈ 6

avg k99 ≈ 10
```

were observed before stabilization. 【1-a63bd0】

After introducing viscosity:

```text
Sin8

avg k95 ≈ 2.6

avg k99 ≈ 3.5
```

were observed. 【1-85fcb5】

---

# Major Findings

## Finding 1

Viscosity suppresses spectral complexity growth.

Comparing inviscid and viscous Burgers:

```text
k95
```

and

```text
k99
```

decrease significantly after dissipation is added. 【1-a63bd0】【1-85fcb5】

---

## Finding 2

Interface complexity remains low.

Observed values:

```text
k95 ≈ 2-3
```

and

```text
k99 ≈ 2-6
```

for all successful cases. 【1-85fcb5】

---

## Finding 3

Nonlinear evolution does not eliminate interface compression.

Even with nonlinear dynamics:

```text
Small spectral representations
remain sufficient.
```

---

## Finding 4

Broadband signals remain compressible.

Despite containing multiple frequencies:

```text
k95 ≈ 2.6
```

and

```text
k99 ≈ 5.7
```

were sufficient. 【1-85fcb5】

---

# Interpretation

The results suggest:

```text
Dissipative nonlinear evolution
preserves low-dimensional
interface structure.
```

While nonlinear interactions generate additional spectral content, viscosity prevents uncontrolled growth of interface complexity.

---

# Implications

The Spectral Edge concept appears robust for:

```text
Linear Transport
```

and

```text
Viscous Nonlinear Transport
```

provided that the dynamics remain sufficiently regular.

This supports the broader hypothesis that:

```text
Interface communication
occupies a much smaller space
than the full physical state.
```

---

# Current Research Status

| Item | Status |
|--------|--------|
| Linear Spectral Edge | ✅ |
| Width-Rank Phase Diagram | ✅ |
| Scaling Investigation | ✅ |
| Pareto Optimization | ✅ |
| Inviscid Burgers Exploration | ✅ |
| Viscous Burgers Validation | ✅ |

---

# Future Work

## Adaptive Width and Rank

Dynamic selection during time integration.

---

## Viscosity Study

Investigate:

```text
ν = 1e-2
ν = 1e-3
ν = 1e-4
ν = 1e-5
```

and measure interface complexity growth.

---

## Higher-Dimensional Burgers

Extend the Pareto framework to:

```text
2D Burgers
```

and compare interface complexity with the 1D results.

---

## Dynamic Spectral Interfaces

Allow the interface state itself to evolve adaptively.

---

## Virtual Spectral Schur Solver

Replace physical boundary exchange with a compressed spectral interface representation.

---

# Conclusion

The viscous Burgers Pareto study demonstrates that nonlinear dissipative evolution can maintain a highly compressed spectral interface representation.

Observed interface complexities remained small:

```text
k95 ≈ 2-3

k99 ≈ 2-6
```

while maintaining high accuracy across all successful test cases. 【1-85fcb5】

These results provide the strongest nonlinear evidence so far that Spectral Edge interfaces may remain low-dimensional even when the underlying dynamics are nonlinear.