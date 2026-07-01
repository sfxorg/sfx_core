# Block 1: Verification Suite

## Objective

Verify that domain decomposition itself introduces no numerical error.

The purpose of this test is to ensure that:

```text
Monolithic Solver
```

and

```text
Decomposed Solver
```

produce identical results when exact interface communication is used.

---

# Configuration

## Domain

```text
2D Advection
```

Grid:

```text
NX = 128
NY = 128
```

Domain split:

```text
Left Panel
Right Panel
```

Interface located at:

```text
x = NX/2
```

---

# Tested Cases

- Constant Field
- Gaussian Packet
- Sinusoidal Packets
- Broadband Packet
- Random Packet
- Oblique Wave

---

# Verification Procedure

For each case:

1. Advance monolithic solution.
2. Advance decomposed solution.
3. Exchange exact interface values.
4. Reconstruct full field.
5. Compare solutions.

---

# Single-Step Verification

A random field was used for a direct stencil verification.

Procedure:

```text
Random Initial Field
→ Monolithic Step
→ Decomposed Step
→ Compare
```

Metric:

```text
Maximum Error
```

Expected:

```text
Machine precision
```

---

# Result

The decomposed solution reproduced the monolithic solution to numerical precision.

Conclusion:

```text
Domain decomposition is verified.
```

---

# Scientific Finding

Any subsequent errors observed in later studies originate from:

- communication compression
- reduced-order transfer
- adaptive truncation

and not from the decomposition framework itself.

---

# Status

✅ PASS