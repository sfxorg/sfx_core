# Spectral Edge Theory Notes

## Purpose

This document records working hypotheses and theoretical interpretations arising from the Spectral Edge experiments.

The goal is to move beyond empirical observations toward mathematical justification suitable for:

- technical papers
- conference publications
- future patent applications

---

# Background

The Spectral Edge framework replaces direct boundary communication with a compressed spectral interface representation.

The central idea is:

```text
Physical Domain A
        ↓
 Spectral Edge
        ↓
Physical Domain B
```

rather than communicating full interface traces.

---

# Fundamental Question

Why can a small interface representation reproduce the behavior of a much larger physical domain?

---

# V3 Experimental Results

## Width-Rank Phase Diagram

| Case | Best Width | Best Rank |
|--------|--------:|--------:|
| Gaussian | 2 | 2 |
| Sin8 | 32 | 4 |
| Sin32 | 8 | 4 |
| Broadband | 32 | 8 |

Observed errors remained near machine precision.

---

# Observation 1

## No Universal Interface Width

Different signals prefer different interface widths.

Examples:

```text
Gaussian
Best Width = 2
```

```text
Sin8
Best Width = 32
```

```text
Sin32
Best Width = 8
```

This suggests that interface geometry is signal dependent.

---

# Hypothesis 1

## Width Represents Spatial Correlation Length

Interpretation:

```text
Width
≈
Spatial Support
```

Localized signals require only local interface information.

Examples:

```text
Gaussian
```

Large-scale coherent structures require larger interface support.

Examples:

```text
Sin8
```

Prediction:

```text
Longer wavelength
→ Larger optimal width
```

---

# Observation 2

## Spectral Rank Remains Small

Observed optimal ranks:

```text
2
4
8
```

even when solution errors were near machine precision.

---

# Hypothesis 2

## Rank Represents Spectral Complexity

Interpretation:

```text
Rank
≈
Interface Information Content
```

Examples:

```text
Gaussian
Rank = 2
```

```text
Broadband
Rank = 8
```

Prediction:

```text
More frequencies
→ Larger rank
```

---

# Observation 3

## Width and Rank Are Independent

The experiments indicate:

```text
Sin8
Width = 32
Rank  = 4
```

and

```text
Sin32
Width = 8
Rank  = 4
```

The optimal rank remains unchanged while the optimal width changes.

---

# Hypothesis 3

Width and rank represent different physical quantities.

### Width

Controls:

```text
Geometric Support
```

### Rank

Controls:

```text
Spectral Complexity
```

This suggests a two-parameter interface manifold.

---

# Observation 4

## Strong Interface Compression

Examples:

```text
Domain Size = 256
```

Best representations:

```text
W = 2
K = 2
```

or

```text
W = 8
K = 4
```

These interface representations are much smaller than the physical state.

---

# Hypothesis 4

## Interface Dimension Is Much Smaller Than Physical Dimension

Interpretation:

```text
Most solution information
does not need to cross
the interface.
```

Only a small subset of information appears necessary for coupling.

This is consistent with the empirical results.

---

# Observation 5

## Signal-Dependent Optimal Operating Points

The V3 study revealed unique:

```text
(width, rank)
```

pairs for different signal classes.

---

# Hypothesis 5

A Spectral Edge possesses an intrinsic operating point.

Possible formulation:

```text
Optimal Interface State

=
f(
signal bandwidth,
signal wavelength,
interface geometry
)
```

---

# Potential Scaling Laws

## Frequency vs Width

Future study:

```text
Sin4
Sin8
Sin16
Sin32
Sin64
```

Goal:

```text
Optimal Width
=
f(Frequency)
```

---

## Bandwidth vs Rank

Future study:

```text
Single Mode
Dual Mode
Multi Mode
Broadband
```

Goal:

```text
Optimal Rank
=
f(Bandwidth)
```

---

# Interface Manifold Hypothesis

The experiments suggest:

```text
Physical Solution Space
```

is large.

However:

```text
Interface Solution Space
```

may occupy a much smaller manifold.

Conceptually:

```text
Full State

↓

Compressed Interface State

↓

Reconstructed Coupling
```

This is a central hypothesis of the Spectral Edge framework.

---

# Potential Patent Direction

Novelty does not arise from:

- FFT
- Schwarz methods
- Domain decomposition

These are established techniques.

Potential novelty arises from:

```text
Compressed Spectral Interface
Representation
```

where communication occurs through:

```text
Adaptive Width
+
Adaptive Rank
```

instead of through full interface traces.

---

# Long-Term Vision

## Virtual Spectral Interface

Interface evolves as an independent object.

```text
Domain A
    ↔
Spectral Edge
    ↔
Domain B
```

---

## Virtual Spectral Schur Method

Future objective:

```text
Solve the interface directly
in compressed spectral space.
```

instead of exchanging physical boundary values.

---

# Open Questions

1.

```text
Why does Sin32 prefer
smaller width than Sin8?
```

2.

```text
Can width be predicted
from wavelength?
```

3.

```text
Can rank be predicted
from bandwidth?
```

4.

```text
Does a universal scaling law exist?
```

5.

```text
Can the interface evolve
as its own dynamical system?
```

---

# Current Working Thesis

The Spectral Edge experiments suggest that:

```text
Interface communication
occupies a significantly smaller
space than the physical solution.
```

Optimal communication appears to require:

```text
Small Width
+
Small Rank
```

whose values depend on signal structure.

This observation motivates future development of:

```text
Virtual Spectral Interfaces
```

and ultimately:

```text
Virtual Spectral Schur Solvers.
```

---

# Scaling Law Experiment

## Study

A frequency sweep was performed using:

```text
Sin2
Sin4
Sin8
Sin16
Sin32
Sin64
```

For each case:

```text
Width Sweep
=
[2,4,8,16,32,64]

Rank Sweep
=
[1,2,4,8,16]
```

The optimal:

```text
Width
```

and

```text
Rank
```

were determined from the minimum reconstruction error.
Based on the scaling experiment results. 【1-5302ee】

---

# Results

| Frequency | Wavelength | Best Width | Best Rank |
|------------|------------:|------------:|------------:|
| 2 | 0.50000 | 2 | 2 |
| 4 | 0.25000 | 64 | 8 |
| 8 | 0.12500 | 64 | 4 |
| 16 | 0.06250 | 16 | 2 |
| 32 | 0.03125 | 8 | 4 |
| 64 | 0.01562 | 4 | 2 |

Observed errors were near machine precision for all frequencies except the lowest-frequency Sin2 case. 【1-5302ee】

---

# Observation

For:

```text
k = 16
```

best width:

```text
W = 16
```

For:

```text
k = 32
```

best width:

```text
W = 8
```

For:

```text
k = 64
```

best width:

```text
W = 4
```

These satisfy:

```text
16 × 16 = 256
32 ×  8 = 256
64 ×  4 = 256
```

for a domain containing:

```text
N = 256
```

grid points. 【1-5302ee】

---

# Scaling Hypothesis

For resolved periodic signals:

```text
W* · k ≈ N
```

or equivalently:

```text
W* ≈ N / k
```

Since:

```text
λ = L / k
```

this can be written as:

```text
W* ≈ λ / Δx
```

where:

```text
Δx
```

is the grid spacing.

---

# Physical Interpretation

The optimal interface width appears to contain approximately:

```text
one wavelength
```

of the dominant signal.

Examples:

```text
k = 16

λ = 16 grid points

W* = 16
```

```text
k = 32

λ = 8 grid points

W* = 8
```

```text
k = 64

λ = 4 grid points

W* = 4
```

The agreement is exact for all three cases tested. 【1-5302ee】

---

# Revised Width Interpretation

Previous hypothesis:

```text
Width
≈
Spatial Correlation Length
```

Updated hypothesis:

```text
Width
≈
Dominant Wavelength
```

for periodic signals.

This interpretation is more precise and is directly supported by the scaling experiment. 【1-5302ee】

---

# Rank Interpretation

No clear frequency-dependent scaling of rank was observed.

Observed optimal ranks:

```text
2
4
8
```

across the entire frequency sweep. 【1-5302ee】

Current working hypothesis:

```text
Rank
≈
Spectral Bandwidth
```

rather than:

```text
Rank
≈
Frequency
```

Additional experiments involving multi-frequency signals are required.

---

# Candidate Scaling Law

Current evidence suggests:

```text
Optimal Width

W*

≈

Dominant Wavelength
```

or:

```text
W* ≈ N / k
```

for periodic signals.

This is the first experimentally supported scaling law produced by the Spectral Edge research program. 【1-5302ee】

---

# Significance

If validated by additional studies, this result provides a principled method for selecting spectral interface widths:

```text
Estimate dominant wavelength

↓

Choose interface width

W ≈ wavelength
```

rather than selecting interface sizes heuristically.

This would represent an important step toward adaptive spectral interfaces and future Virtual Spectral Schur methods. 【1-5302ee】

---

# Scaling Law Validation Update

## Status

```text
IN PROGRESS
```

The initial scaling experiment suggested:

```text
W* · k ≈ N
```

for several sinusoidal test cases.

A larger validation study was subsequently performed using:

```text
N = 128
N = 256
N = 512
```

and

```text
k = 2
4
6
8
12
16
24
32
48
64
```

to determine whether the observed relationship remained valid across multiple grid sizes and frequencies. 【1-8f1b9c】

---

# Initial Hypothesis

The preliminary hypothesis was:

```text
W* ≈ N / k
```

or equivalently:

```text
W* ≈ λ / Δx
```

where:

```text
λ = wavelength
Δx = grid spacing
```

This interpretation was motivated by earlier observations including:

```text
k = 16  →  W* = 16
k = 32  →  W* = 8
k = 64  →  W* = 4
```

for:

```text
N = 256
```

which satisfy:

```text
W* · k = 256
```

exactly. 【1-f33b4d】

---

# Expanded Validation Results

Additional frequencies and grid sizes revealed more complicated behavior.

Examples:

```text
N = 256

k = 12  →  W* = 64

k = 24  →  W* = 64
```

while:

```text
k = 16  →  W* = 16

k = 32  →  W* = 8

k = 64  →  W* = 4
```

continued to follow the original scaling relationship. 【1-8f1b9c】

---

# Power-Law Fitting

A power-law model:

```text
W = C k^(-α)
```

was fitted independently for each grid.

Estimated exponents:

```text
N = 128
α ≈ 0.539
```

```text
N = 256
α ≈ -0.099
```

```text
N = 512
α ≈ -0.836
```

The fitted exponents are not consistent across grids. 【1-8f1b9c】

---

# Revised Assessment

Current evidence is insufficient to support a universal scaling law of the form:

```text
W* ∝ 1/k
```

The observed behavior appears more complicated than originally expected. 【1-8f1b9c】

---

# Possible Explanation

The current optimization criterion is:

```text
Choose (W,K)
that minimizes error.
```

This may not be the most physically meaningful objective.

Many configurations achieve:

```text
near machine-precision accuracy
```

simultaneously.

As a result:

```text
multiple nearly equivalent minima
```

may exist in the width-rank landscape. 【1-8f1b9c】

---

# Alternative Selection Principle

Future studies should consider:

```text
Choose smallest width
subject to

error < tolerance
```

for example:

```text
error < 1e-10
```

This would prioritize compression efficiency rather than absolute error minimization.

---

# Current Working Conclusion

The original scaling hypothesis:

```text
W* · k ≈ N
```

remains an interesting observation for selected frequencies.

However:

```text
the expanded validation study
does not yet support it as a
universal scaling law.
```

Additional investigation is required. 【1-8f1b9c】

---

# Scientific Value

This negative result is important.

The validation study demonstrates that:

```text
the apparent scaling law
survived limited testing
but did not generalize
under broader conditions.
```

This eliminates an overly simplistic interpretation and helps refine future theoretical development. 【1-8f1b9c】

---

# Updated Status of Scaling Hypothesis

| Hypothesis | Status |
|------------|--------|
| W* ≈ N/k | ⚠️ Not yet validated |
| W* ≈ wavelength | ⚠️ Partially supported |
| Width depends on signal structure | ✅ Supported |
| Rank remains small | ✅ Supported |
| Interface compression exists | ✅ Supported |

---

# Next Research Question

Rather than asking:

```text
What width minimizes error?
```

future work should ask:

```text
What is the smallest width
that achieves a target accuracy?
```

This may provide a more physically meaningful route toward adaptive spectral interfaces and future Virtual Spectral Schur methods.
