# Spectral Edge V3: Width-Rank Phase Diagram Study

## Date

July 2026

---

# Objective

Determine the optimal combination of:

```text
Interface Width
```

and

```text
Spectral Rank
```

for virtual spectral edge coupling.

The study investigates whether interface information can be represented using:

- small geometric support
- small spectral complexity
- high solution accuracy

simultaneously.

---

# Configuration

## Domain

```text
1D Linear Advection
```

---

## Grid

```text
N = 256
```

Domain split into:

```text
Left Domain
Right Domain
```

at:

```text
N/2
```

---

## Time Integration

```text
RK4
```

---

## Study Parameters

### Width Sweep

```text
W = 2
W = 4
W = 8
W = 16
W = 32
```

### Rank Sweep

```text
K = 1
K = 2
K = 4
K = 8
K = 16
```

---

# Test Cases

## Gaussian

```python
exp(-(x-x0)^2/sigma^2)
```

Localized smooth packet.

---

## Sin8

```python
sin(2*pi*8*x)
```

Low-frequency periodic signal.

---

## Sin32

```python
sin(2*pi*32*x)
```

Higher-frequency periodic signal.

---

## Broadband

```python
sin(8x)
+
0.5 sin(16x)
+
0.25 sin(32x)
```

Multi-frequency signal.

---

# Diagnostics

For every pair:

```text
(Width, Rank)
```

the following were measured:

- Maximum Error
- Average k95
- Average k99

---

# Definition of k95

```text
k95
```

denotes the minimum number of spectral modes required to capture:

```text
95%
```

of interface energy.

---

# Definition of k99

```text
k99
```

denotes the minimum number of spectral modes required to capture:

```text
99%
```

of interface energy.

---

# Results

## Gaussian

### Best Configuration

```text
Width = 2

Rank = 2
```

### Error

```text
7.789e-11
```

### Interpretation

The localized Gaussian packet requires only:

```text
2 interface points
2 spectral modes
```

to achieve near machine-precision accuracy.

This indicates that smooth localized solutions occupy an extremely compact interface representation.

---

# Sin8

### Best Configuration

```text
Width = 32

Rank = 4
```

### Error

```text
1.543e-14
```

### Interpretation

The low-frequency sinusoidal signal prefers:

```text
large geometric support
```

while maintaining:

```text
very small spectral rank
```

This suggests that width and spectral complexity represent different properties.

---

# Sin32

### Best Configuration

```text
Width = 8

Rank = 4
```

### Error

```text
1.565e-14
```

### Interpretation

The higher-frequency wave requires:

```text
smaller width
```

than Sin8 while retaining the same rank.

This was a surprising result.

It suggests that optimal interface width is signal dependent.

---

# Broadband

### Best Configuration

```text
Width = 32

Rank = 8
```

### Error

```text
2.276e-14
```

### Interpretation

Broadband signals required:

```text
larger width
```

and

```text
larger rank
```

than single-mode signals.

This result is physically intuitive because the interface must represent multiple frequencies simultaneously.

---

# Summary Table

| Case | Best Width | Best Rank | Error |
|--------|--------:|--------:|--------:|
| Gaussian | 2 | 2 | 7.789e-11 |
| Sin8 | 32 | 4 | 1.543e-14 |
| Sin32 | 8 | 4 | 1.565e-14 |
| Broadband | 32 | 8 | 2.276e-14 |

---

# Scientific Findings

## Finding 1

No universal interface width exists.

Different signals prefer different geometric interface supports.

---

## Finding 2

Optimal spectral ranks remain small.

Observed optimal ranks:

```text
2
4
8
```

even when accurate reconstruction approaches machine precision.

---

## Finding 3

Width and rank are independent quantities.

Evidence suggests:

```text
Width
```

controls:

```text
Geometric Support
```

while:

```text
Rank
```

controls:

```text
Spectral Complexity
```

---

## Finding 4

The interface representation remains strongly compressed.

Examples:

```text
Gaussian
W=2
K=2
```

```text
Sin32
W=8
K=4
```

These are dramatically smaller than the full solution state.

---

# Implications

The results support the hypothesis that:

```text
Interface spaces are much smaller
than physical solution spaces.
```

This observation motivates:

```text
Virtual Spectral Interfaces
```

and eventually:

```text
Virtual Spectral Schur Methods
```

where interfaces evolve using compact spectral representations rather than full-state communication.

---

# Current Status

| Item | Status |
|--------|--------|
| Spectral Edge Coupling | ✅ Working |
| Width Sweep | ✅ Completed |
| Rank Sweep | ✅ Completed |
| Width-Rank Phase Diagram | ✅ Completed |
| Signal-Dependent Optima Found | ✅ Confirmed |
| Interface Compression Demonstrated | ✅ Confirmed |

---

# Future Work

## V4

Adaptive Width and Adaptive Rank

Goal:

```text
Automatically select

minimum width
+
minimum rank

for a target error tolerance.
```

---

## V5

Dynamic Spectral Interface

Investigate whether the interface itself can evolve as an independent spectral object.

---

## Long-Term Goal

```text
Virtual Spectral Schur Solver
```

in which communication occurs primarily through a compressed spectral interface state rather than through full boundary traces.

---

# Conclusion

The V3 Width-Rank Phase Diagram demonstrates that spectral interfaces possess signal-dependent optimal operating points while requiring only a small number of spectral modes.

The results provide strong evidence that interface communication can be represented in a highly compressed form and establish an experimental foundation for future virtual spectral interface and Schur-based methods.