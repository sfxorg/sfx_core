# Block 3: Adaptive Spectral Schwarz

## Objective

Replace fixed communication ranks with adaptive ranks.

Instead of:

```text
Rank = Fixed
```

determine:

```text
Minimum rank
required
for a target tolerance.
```

---

# Configuration

Grid:

```text
128 × 128
```

Cases:

- Gaussian
- Sin8
- Sin32
- Broadband
- Random

Adaptive tolerances:

```text
1e-1
1e-2
1e-3
1e-4
1e-5
```

---

# FFT Adaptive Communication

Procedure:

```text
Compute Interface Spectrum
↓
Determine Energy Content
↓
Retain Smallest Rank
meeting Tolerance
↓
Transfer
```

---

# POD Adaptive Communication

Procedure:

```text
Project Interface Trace
onto POD Basis
↓
Determine Required Rank
↓
Transfer
```

Several POD versions were developed:

- POD v1
- POD v2
- POD v3
- POD v4
- POD v5

---

# POD Development History

## v1

Single-case training.

Result:

```text
Poor generalization.
```

---

## v2

Multi-case training.

Result:

```text
Improved robustness.
```

---

## v3

Modified normalization strategy.

Result:

```text
Improved convergence.
```

---

## v4

Balanced frequency training.

Added:

```text
Sin64
Sin96
```

Result:

```text
Better basis coverage.
```

---

## v5

Singular-value energy rank selection.

Result:

```text
Most stable POD variant.
```

---

# Final Results

## FFT

Passed all benchmark cases:

- Gaussian
- Sin8
- Sin32
- Broadband
- Random

Observed:

```text
Tolerance ↓
Error ↓
```

with smooth convergence.

---

## POD

Successful for:

- Gaussian
- Sin8
- Broadband
- Random

Persistent difficulty:

```text
Sin32
```

High-frequency interface content remained challenging.

---

# Primary Scientific Finding

FFT-based adaptive communication is highly robust.

The method accurately compresses interface information across a wide range of spectral content while maintaining solution accuracy.

---

# Secondary Scientific Finding

POD performance depends strongly on:

- basis quality
- training coverage
- retained rank

High-frequency interfaces require substantially richer representations.

---

# Conclusion

Adaptive FFT communication emerged as the most reliable communication-compression strategy.

This establishes the foundation of:

```text
Adaptive Spectral Schwarz
```

for decomposed PDE solvers.

---

# Status

✅ FFT VERIFIED

⚠️ POD PARTIALLY VERIFIED

✅ Schwarz Framework VERIFIED