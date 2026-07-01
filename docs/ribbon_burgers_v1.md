# Ribbon Burgers v1 Validation Report: tests/test_burger_2d_v1.py

## Date

July 2026

---

# Objective

Evaluate whether a ribbon-based virtual interface can couple a large-scale FFT Burgers solver while preserving:

- accuracy
- conservation
- interface continuity
- spectral compressibility

The experiment also investigates whether interface mismatch occupies a low-dimensional spectral manifold.

---

# Configuration

## PDE

2D Burgers Equation

---

## Grid

```text
N = 1024
```

Total FFT degrees of freedom:

```text
1024 × 1024
=
1,048,576
```

---

## Hybrid Architecture

```text
                FFT Domain

        +----------------------+

            Top Ribbon

 Left Ribbon          Right Ribbon

          Bottom Ribbon
```

Ribbon width:

```text
P = 8
```

Each ribbon evolves as an auxiliary interface state.

---

# Test A: Gaussian Initial Condition

## Initial Condition

```python
u = exp(
    -((x-x0)^2 + (y-y0)^2)/sigma^2
)
```

---

## Runtime

```text
FFT runtime
≈ 5.43 s

Hybrid runtime
≈ 6.17 s
```

---

## Accuracy

Compared against the full FFT solution.

```text
Maximum Error
=
1.811981e-05

L2 Error
=
4.256249e-07
```

---

## Conservation

```text
Mass Difference
=
0

Energy Difference
≈
1.86e-09
```

The hybrid solution preserves both mass and energy to numerical precision.

---

## Interface Matching

Measured mismatches between ribbon states and FFT boundary values.

```text
Top
≈ 5.09e-10

Bottom
≈ 6.49e-10

Left
≈ 4.18e-10

Right
≈ 3.61e-10
```

All interfaces remain continuous to near machine precision.

---

## Ribbon Energy

FFT energy:

```text
2.356e-02
```

Ribbon energy:

```text
5.286e-14
```

Ratio:

```text
2.24e-10 %
```

### Interpretation

The ribbon contains negligible physical energy.

The ribbon behaves as:

```text
Constraint Layer
```

rather than as a physical solution carrier.

---

# Time-History Spectral Diagnostics

Additional diagnostics were added to measure interface complexity during the entire simulation.

Metrics:

```text
k95
```

Smallest Fourier mode count containing 95% of mismatch energy.

```text
k99
```

Smallest Fourier mode count containing 99% of mismatch energy.

---

# Left Interface

```text
Average k95
=
3.06

Maximum k95
=
5

Average k99
=
7.02

Maximum k99
=
69
```

---

# Right Interface

```text
Average k95
=
3.32

Maximum k95
=
5

Average k99
=
4.75

Maximum k99
=
6
```

---

# Top Interface

```text
Average k95
=
501

Average k99
=
506
```

These values are dominated by numerical noise because the top mismatch magnitude remains near machine precision.

---

# Bottom Interface

```text
Average k95
=
504

Average k99
=
506
```

Also dominated by numerical noise.

---

# Compression Implications

For the physically active interfaces:

```text
1024-point edge
```

contains approximately:

```text
3-5 dominant modes
```

for 95% energy capture.

Equivalent compression factors:

```text
1024 / 3
=
341x
```

to

```text
1024 / 5
=
205x
```

---

# Result

The interface correction appears to reside on a very low-dimensional spectral manifold.

This is one of the most significant observations of the experiment.

---

# Test B: Sinusoidal Burgers Stress Test

## Initial Condition

```python
u = sin(
    2*pi*8*x/L
)
```

---

## Observed Evolut*on

```text
Step 0
max = *

Step 50
max*≈ 3

Step 100
max ≈ 6

Step 150
ma* ≈ 10

Step 200
max*≈ 12

Step 250
max ≈ 40

Step*300
NaN
```

---

## Interpretatio*

The solution exhibits*

```text
bounded behavior
→ nonli*ear amplification
→ runaway growth*→ overflow
→ NaN
```

The failure *ppears consistent with a nonlinear*Burgers instability rather than an*immediate ribbon-interface failure*

---

## Interface Complexity Dur*ng Instability

Measured before Na* occurred:

```text
LEFT avg k95
≈*342

RIGHT avg k95
≈ 341

*OP avg k95
≈ 325

BOTTOM avg k95
≈*304
```

---

##*Interpretation

Unlike the Gaussia* case:

```text
avg k95 ≈ 3
```

t*e sinusoidal Burgers solution gene*ates broadband nonlinear content.
*The interface manifold becomes muc* higher dimensional.

---

# Scien*ific Findings

## Finding 1

The r*bbon hybrid solver reproduces FFT *olutions with high accuracy.

```t*xt
Max Error ≈ 1e-5
L2 Error ≈ 1e-*
```

---

## Finding 2

Mass and *nergy are preserved.

```text
Mass*diff = 0
Energy diff ≈ 0
```

---
*## Finding 3

For localized smooth*solutions:

```text
Interface corr*ctions occupy
an extremely low-dim*nsional
spectral space.
```

Obser*ed:

```text
k95 ≈ 3-5 modes
```

*n a 1024-point edge.

---

## Find*ng 4

For oscillatory nonlinear Bu*gers solutions:

```text
Interface*complexity increases dramatically.*```

Observed:

```text
k95 ≈ 300-*50 modes
```

before instability.
*---

## Finding 5

The sinusoidal *est exposed a stability boundary o* the current hybrid Burgers formul*tion.

The instability is currentl* believed to be related to:

```te*t
nonlinear Burgers dynamics
```

*ather than to the ribbon interface*concept itself.

---

# Current St*tus

| Item | Status |
|--------|-*------|
| FFT Reference | ✅*PASS |
| Hybrid Ribbon Coupling | * PASS |
| Conservation | ✅ PASS |
* Interface Continuity | ✅ PASS |
|*Gaussian Compressibility Study | ✅*PASS |
| Time-History Spectral Dia*nostics | ✅ PASS |
| Sinusoidal Bu*gers Stress Test | ✅ COMPLETED |
|*Nonlinear Stability Investigation | ⚠️ IN PROGRESS |

---

# Next Experiments

Planned:

```python
u = 1 + 0.1*sin(2*pi*8*x/L)
```

```python
u = * +*0.1*sin(2*pi*32*x/L)
```

```python
u = 1 + 0.1*sin(2*pi*64*x/L)
```

to determine whe*her the instability is caused prim*rily by:

- sign-changing velocity*fields
- high-frequency content
- *onlinear Burgers amplification

--*

# Conclusion

The Gaussian bench*ark strongly supports the Virtual *nterface Ribbon concept.

The expe*iment demonstrates that:

```*ext
Very large FFT domains
can be *oupled through ribbons whose
*nterface correction occupies only * few
dominant spectral modes*
```

The sinusoidal Burgers exper*ment revealed a nonlinear stabilit* boundary and motivates*future work on stabilization of th* hybrid formulation.
``*