# Spectral Edge Pareto Study

## Date

July 2026

---

# Objective

Previous studies selected interface parameters using:

```text
Minimum Error
```

However, this approach often favored larger interface representations even when smaller interfaces achieved essentially identical accuracy.

The objective of this study is to identify:

```text
Minimum Interface Cost

subject to

Error < Tolerance
```

where interface cost is defined as:

```text
Cost = Width × Rank
```

---

# Motivation

The fundamental question is not:

```text
What interface produces the smallest error?
```

The fundamental question is:

```text
What is the smallest interface
that achieves a desired accuracy?
```

This reframes the problem as an interface compression problem.

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

---

## Parameter Sweep

### Widths

```text
2
4
8
16
32
64
```

### Ranks

```text
1
2
4
8
16
```

### Tolerances

```text
1e-4
1e-6
1e-8
1e-10
```

---

# Test Cases

## Gaussian

Localized smooth packet.

---

## Sin8

Single low-frequency mode.

---

## Sin32

Single higher-frequency mode.

---

## Broadband

Multi-frequency signal.

---

# Cost Definition

Interface complexity is measured using:

```text
Cost = Width × Rank
```

Examples:

```text
W = 8
K = 2

Cost = 16
```

```text
W = 32
K = 8

Cost = 256
```

Smaller cost corresponds to a smaller interface representation.

---

# Pareto Frontier

Each experiment produces:

```text
(Error, Cost)
```

pairs.

A configuration is considered Pareto-optimal when:

```text
No other configuration
has both

smaller error

and

smaller cost.
```

This identifies the most efficient interface representations.

---

# Results

## Sin8

### Optimal Interface

```text
Width = 32

Rank = 2

Cost = 64
```

### Error

```text
1.898e-14
```

The same optimal configuration satisfied all tested tolerances.

---

## Sin32

### Optimal Interface

```text
Width = 8

Rank = 2

Cost = 16
```

### Error

```text
1.926e-14
```

A remarkably small interface representation reproduced the solution to near machine precision.

---

## Broadband

### Optimal Interface

```text
Width = 32

Rank = 8

Cost = 256
```

### Error

```text
2.276e-14
```

The increase in rank is consistent with the larger information content of the signal.

---

## Gaussian

### Accuracy 1e-4 to 1e-8

```text
Width = 2

Rank = 1

Cost = 2
```

### Error

```text
1.301e-10
```

### Accuracy 1e-10

```text
Width = 2

Rank = 2

Cost = 4
```

### Error

```text
7.789e-11
```

This represents the most compressed interface observed in the study.

---

# Summary Table

| Case | Width | Rank | Cost | Error |
|--------|--------:|--------:|--------:|--------:|
| Gaussian | 2 | 1 | 2 | 1.301e-10 |
| Gaussian (1e-10) | 2 | 2 | 4 | 7.789e-11 |
| Sin8 | 32 | 2 | 64 | 1.898e-14 |
| Sin32 | 8 | 2 | 16 | 1.926e-14 |
| Broadband | 32 | 8 | 256 | 2.276e-14 |

Results obtained from the Pareto analysis study. 【1-d4059f】

---

# Major Findings

## Finding 1

Interface selection should be performed using:

```text
Minimum Cost

subject to

Accuracy Constraint
```

rather than using:

```text
Minimum Error
```

alone.

---

## Finding 2

Optimal ranks are smaller than previous phase-diagram estimates.

Examples:

```text
Sin8:
K = 2
```

```text
Sin32:
K = 2
```

Previous studies suggested larger ranks when error minimization alone was used.

---

## Finding 3

Interface representations remain extremely compact.

Examples:

```text
Domain Size = 256
```

```text
Gaussian Cost = 2
```

```text
Sin32 Cost = 16
```

These interfaces are dramatically smaller than the full solution state.

---

## Finding 4

Signal complexity directly affects interface cost.

Ordering:

```text
Gaussian
↓
Sin32
↓
Sin8
↓
Broadband
```

corresponds to increasing information content.

---

# Interpretation

The Pareto study suggests that:

```text
Interface communication
occupies a much lower-dimensional
space than the physical solution.
```

High-fidelity coupling can be achieved using:

```text
Small Width
+
Small Rank
```

selected according to a user-defined accuracy target.

---

# Comparison to Scaling Studies

Previous work attempted to derive:

```text
Width = f(Frequency)
```

relationships.

The Pareto study indicates that a more important quantity may be:

```text
Minimum Cost
=
f(Accuracy,
Signal Structure)
```

rather than a simple frequency scaling law.

---

# Practical Rule

For a target tolerance:

```text
1.
Sweep candidate Widths

2.
Sweep candidate Ranks

3.
Choose the smallest
(W × K)

satisfying the
accuracy requirement
```

This provides an adaptive interface-selection strategy.

---

# Implications

The Pareto approach provides a direct path toward:

```text
Adaptive Spectral Interfaces
```

where communication size is determined automatically by accuracy requirements.

This observation is likely more important than any current width-scaling relation.

---

# Current Status

| Item | Status |
|--------|--------|
| Width-Rank Phase Diagram | ✅ |
| Scaling Investigation | ✅ |
| Pareto Frontier Analysis | ✅ |
| Cost-Constrained Interface Selection | ✅ |
| Interface Compression Demonstrated | ✅ |

---

# Future Work

## Adaptive Online Selection

Select:

```text
Width
+
Rank
```

automatically during time integration.

---

## Nonlinear Burgers Study

Apply Pareto interface selection to:

```text
1D Burgers

2D Burgers
```

and measure:

```text
Adaptive Cost
vs
Accuracy
```

---

## Virtual Spectral Interface

Represent the interface as a dynamically evolving compressed state.

---

## Virtual Spectral Schur Solver

Solve inter-domain communication directly within the compressed spectral representation.

---

# Conclusion

The Pareto study demonstrates that the most effective interface representation is not obtained by minimizing error alone.

Instead:

```text
Minimum-Cost Interface

subject to

Target Accuracy
```

produces highly compressed interface states while maintaining near machine-precision accuracy.

These results provide the strongest support so far for the Spectral Edge concept and establish a practical framework for adaptive compressed interface communication.