# Block 2: Fixed-Rank Compression Suite

## Objective

Evaluate interface communication compression using fixed spectral ranks.

Two communication strategies were investigated:

```text
FFT Compression
```

and

```text
POD Compression
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

Communication ranks:

```text
1
2
4
8
16
32
64
```

---

# Method

For every timestep:

```text
Extract Interface Trace
↓
Compress
↓
Transfer
↓
Reconstruct
↓
Continue Solve
```

---

# FFT Compression

Communication performed using:

```text
Discrete Fourier Transform
```

Only a specified number of modes are retained.

---

# POD Compression

Communication performed using:

```text
Proper Orthogonal Decomposition
```

A reduced basis was constructed from training snapshots.

---

# Results

## FFT

Observed behavior:

```text
Rank ↑
Error ↓
```

for every test case.

FFT exhibited:

- monotonic convergence
- smooth error reduction
- robustness to oscillatory content

---

## POD

Observed behavior:

```text
Rank ↑
Error ↓
```

for most test cases.

POD performance depended on the representativeness of the training basis.

---

# Scientific Findings

## FFT

Robust across:

- smooth fields
- oscillatory fields
- broadband fields
- random fields

---

## POD

Effective when:

```text
Interface behavior
≈
Training behavior
```

Performance degraded for strongly out-of-distribution content.

---

# Conclusion

Both FFT and POD are viable interface compression mechanisms.

FFT demonstrated greater robustness.

---

# Status

✅ PASS