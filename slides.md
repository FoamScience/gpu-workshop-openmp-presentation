---
theme: foamscience
hideInToc: true
title: OpenMP Target Offloading Analysis - OpenFOAM_HMM
class: text-center
drawings:
  persist: false
transition: slide-left
mdc: true
duration: 35min
logoLight: https://foamscience.github.io/bayesian-optimization-for-combustion/images/nhr-tu-logo.png
logoDark: https://foamscience.github.io/bayesian-optimization-for-combustion/images/nhr-tu-logo-dark.png
layout: cover
background: https://foamscience.github.io/bayesian-optimization-for-combustion/images/cover.jpg
bachgroundOpacity: 0.50
footer:
  author: Mohammed Elwardi Fadeli
  affiliation: IANUS - Jan. 2026
---

# OpenMP Target Offloading Analysis

A dive into OpenFOAM_HMM repository

---
layout: two-cols
---

# The Story

<br>

<Toc :columns="2" :mode="onlyCurrentTree" />

---

# Overview

A comprehensive GPU-porting effort for OpenFOAM using **OpenMP target offloading** by ROCm team

- Designed primarily for **AMD GPUs** (MI100/MI200/MI300 series)
- Leverages **Unified Shared Memory (USM)** via HMM (Heterogeneous Memory Management)
- Explicit device memory management for performance-critical kernels

<br>


| Metric | Count |
|--------|-------|
| `#pragma omp target` directives | ~160 |
| Files with target offloading | ~40 |
| `omp target alloc` usage | Rare |
| `declare target` kernel regions | 3 |


---

# Library Components Using OMPTO

<br>

**Core Matrix Classes** (`src/OpenFOAM/matrices/lduMatrix/`)

The linear algebra subsystem is the most heavily GPU-accelerated component (simple loop offloads):

- **Matrix-Vector Operations** - Sparse matrix multiplication (Amul, Tmul, residual, H1, sumA)
- **PCG Solver** - Preconditioned Conjugate Gradient
- **PBiCGStab Solver** - Stabilized Bi-Conjugate Gradient
- **GAMG Solver** - Geometric-Algebraic Multi-Grid (scale, agglomerate, solve)
- **Gauss-Seidel Smoother** - Iterative smoother (parallelized)
- **DIC/FDIC Smoother** - Diagonal Incomplete Cholesky
- **Diagonal Preconditioner** - Simple diagonal preconditioning
- **DILU Preconditioner** - Diagonal Incomplete LU

---

# Library Components Using OMPTO

<br>

**Finite Volume Framework** (`src/finiteVolume/`)

- Gradient schemes (especially the limited ones)
- Some interpolation schemes
- Surface integration

<br>

**Core Field Operations** (`src/OpenFOAM/fields/`)

- **Scalar Field** - Scalar field operations
- **Field Template** - Generic field operations
- **FieldField** - Field of fields operations

---

# Simple Loop Offloading

~80% of cases

The majority of GPU-offloaded code consists of straightforward parallel loops where the original algorithm is preserved.

<br>

**Example: Diagonal Matrix Initialization** (`lduMatrixATmul.C:198-202`)

```cpp {all|1|2-6}
#pragma omp target teams distribute parallel for if(nCells>TARGET_CUT_OFF)
for (label cell=0; cell<nCells; cell++)
{
    ApsiPtr[cell] = diagPtr[cell]*psiPtr[cell];
}
```


> They basically hunted for loops which have **no data dependencies**; or those which are **easy to break**.

---

# Significant Algorithm Modifications

~20% of cases

Some algorithms required fundamental restructuring to enable GPU parallelization.

## Gauss-Seidel Smoother → Parallel Jacobi-Like Iteration

Cell updates affects neighbors immediately:

```cpp {all|4-6|7-9}
for (label celli=0; celli<nCells; celli++)
{
    fStart = fEnd; fEnd = ownStartPtr[celli + 1]; psii = bPrimePtr[celli];
    // Uses already-updated psi values from previous cells
    for (label facei=fStart; facei<fEnd; facei++) psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
    psii /= diagPtr[celli];
    // Immediately updates neighbors - creates sequential dependency
    for (label facei=fStart; facei<fEnd; facei++) bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
    psiPtr[celli] = psii;
}
```

---

# Significant Algorithm Modifications

The algorithm was restructured into independent phases that can be parallelized:

```cpp
// Temporary arrays for parallel computation; GaussSeidelSmoother.C:172-312
solveScalarField Z(psi.size());  /* correction */     solveScalarField R(psi.size());  /* residual */
```

**PHASE 1: Compute $r = A*u$**

```cpp {all|1-2|4-15}
#pragma omp target teams distribute parallel for if(nCells > 5000)
for (label celli=0; celli<nCells; ++celli) r_ptr[celli] = diagPtr[celli]*u_ptr[celli];

#pragma omp target teams distribute parallel for if(nFaces > 5000) thread_limit(256)
for (label face=0; face<nFaces; face+=2) {
    const label nf = (nFaces-face) > 1 ? 2 : 1;
    #pragma unroll 2
    for (label i = 0; i < nf; ++i){
        #pragma omp atomic
        r_ptr[uPtr[face+i]] += lowerPtr[face+i]*u_ptr[lPtr[face+i]];
        #pragma omp atomic
        r_ptr[lPtr[face+i]] += upperPtr[face+i]*u_ptr[uPtr[face+i]];
    }
}
```

---

# Significant Algorithm Modifications

**PHASE 2: Compute correction z = (rhs - r)/D and update u**

```cpp
#pragma omp target teams distribute parallel for if(nCells > 3000)
for (label celli=0; celli<nCells; celli++) {
    scalar r = rhs_ptr[celli] - r_ptr[celli];
    z_ptr[celli] = r / diagPtr[celli];
    u_ptr[celli] += z_ptr[celli];
}
```

**PHASE 3: Additional correction sweeps (upper triangular only)**

```cpp
scalar multiplier = -1.0;
for (label sweepID = 0; sweepID < max_sweeps; sweepID++) {
    #pragma omp target teams distribute parallel for if(nCells > 3000)
    for (label celli=0; celli<nCells; celli++) {
        // r = U * z (can be parallelized per-row)
        // ...
    }
    multiplier *= -1.0;
}
```

---

# Key Changes in Algorithm Restructuring

<br>

- Introducing **temporary arrays (Z, R)** to break data dependencies
- Level scheduling to break data dependencies -> mesh dependent!
- **Multi-sweep approach** approximates original convergence behavior

<br>

CSR Matrix Format for Efficient Row Access seems **necessary**
- Idea: each thread/task processes one row independenctly
- LDU not very GPU friendly


---

# Code Patterns

## Unified Shared Memory Declaration

Every file with target offloading includes this guard:

```cpp
#ifdef USE_OMP
#include <omp.h>
  #ifndef OMP_UNIFIED_MEMORY_REQUIRED
  #pragma omp requires unified_shared_memory
  #define OMP_UNIFIED_MEMORY_REQUIRED
  #endif
#endif
```


<br>

This enables **transparent memory access** between CPU and GPU via HMM.

---

# Code Patterns


## Standard Target Directive Pattern

```cpp
#pragma omp target teams distribute parallel for [thread_limit(N)] if(size > threshold)
for (label i = 0; i < size; i++)
{
    // kernel body
}
```

<br>

## AtomicAccumulator Wrapper Class

A custom C++ wrapper provides type-safe atomic accumulation:

**File:** `src/OpenFOAM/primitives/AtomicAccumulator/AtomicAccumulator.H`

Basically `std::atomic_ref` but with custom focus to faces-to-cell writes.

```cpp
atomicAccumulator(ivf[owner[facei+i]]) += issf[facei+i];
atomicAccumulator(ivf[neighbour[facei+i]]) -= issf[facei+i];
```

---

## Loop Unrolling for Instruction-Level Parallelism

Manual 2x or 4x unrolling is used extensively:

```cpp {all|2|4-5|8-13}
#pragma omp target teams distribute parallel for thread_limit(256) if(nCells>TARGET_CUT_OFF)
for (label face=0; face<nFaces; face+=2) {
    const label nf = (nFaces-face) > 1 ? 2 : 1;
    #pragma unroll 2
    for (label i = 0; i < nf; ++i){
        const label l_val = lPtr[face+i];
        const label u_val = uPtr[face+i];
        #pragma omp atomic
        ApsiPtr[u_val] += lowerPtr[face+i]*psiPtr[l_val];
        #pragma omp atomic
        ApsiPtr[l_val] += upperPtr[face+i]*psiPtr[u_val];
    }
}
```

---

## Declare Target for Device Functions/Kernels

Functions callable from target regions:

```cpp
// faceLimitedGrad.H:152-184
#pragma omp declare target
template<>
inline void faceLimitedGrad<double>::limitFace
(
    double& limiter,
    const double maxDelta,
    const double minDelta,
    const double extrapolate
) const
{
    // implementation
}
#pragma omp end declare target
```

---

## Reduction Pattern

```cpp {all|1|3-6|7-11}
solveScalar scalingFactorNum = 0.0, scalingFactorDenom = 0.0;

#pragma omp target teams distribute parallel for \
    reduction(+:scalingFactorNum, scalingFactorDenom) \
    map(tofrom:scalingFactorNum,scalingFactorDenom) \
    if(nCells>3000)
for (label i=0; i<nCells; i++)
{
    scalingFactorNum += fieldPtr[i]*sourcePtr[i];
    scalingFactorDenom += fieldPtr[i]*AcfPtr[i];
}
```

**File:** `GAMGSolverScale.C:74-84`

---

# Data Movement Strategies

## Primary Strategy: Unified Shared Memory (USM)

The repository relies heavily on AMD's HMM (Heterogeneous Memory Management):

```cpp
#pragma omp requires unified_shared_memory
```

<br>

- No explicit `map()`, `to()`, `from()`, `tofrom()` clauses needed
- Runtime handles memory coherency automatically
- Simplifies porting but may have performance implications for non-coherent access patterns

---

# Data Movement Strategies

## Explicit Memory Management

For performance-critical kernels, explicit allocation is used.

**File:** `lduMatrixATmul.C:385-444`

## Explicit Map Clause (Also Rare)

Only used for reductions:

```cpp
#pragma omp target teams distribute parallel for \
    reduction(+:scalingFactorNum, scalingFactorDenom) \
    map(tofrom:scalingFactorNum,scalingFactorDenom) \
    if(nCells>3000)
```

---

# AMD GPU-Specific Optimizations

Fast Floating-Point Atomics

```cpp
#pragma omp atomic hint(AMD_fast_fp_atomics)
ApsiPtr_work_array[uPtr[face]] += lowerPtr[face]*psiPtr[lPtr[face]];
```

ROCm Profiling Integration (roctracer)

```cpp
#ifdef USE_ROCTX
#include <roctracer/roctx.h>
#endif

// Usage throughout the code:
#ifdef USE_ROCTX
roctxRangePush("lduMatrix::Amul");
#endif
// ... GPU kernels ...
#ifdef USE_ROCTX
roctxRangePop();
#endif
```

---

# Precision handling & thread limits

<br>

Thread limits are tuned for AMD GPU architecture - **64, 128, and 256** are enforced throughout the code.

```cpp
// lduMatrixATmul.C:48-57
#if defined(WM_SP)
#define _FP_TYPE_scalar float
#define _FP_TYPE_solve_scalar float
#elif defined(WM_SPDP)
#define _FP_TYPE_scalar float
#define _FP_TYPE_solve_scalar double
#elif defined(WM_DP)
#define _FP_TYPE_scalar double
#define _FP_TYPE_solve_scalar double
#endif
```

---

# A quick Experiment

<br>

<br>

<div class="text-center">

Potential offloading of ILUC0 preconditioner (Foam-Extend, block-matrices) with OpenMP

<div class="text-3xl text-center"> <mdi-dot />Too embarrassing to show here<mdi-dot /> </div>

</div>
