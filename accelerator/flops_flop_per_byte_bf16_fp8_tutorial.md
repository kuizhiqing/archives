# FLOPS, FLOP/Byte, and Why BF16 and FP8 Behave Differently

## 1. Overview

When comparing GPU performance for deep learning workloads, two metrics are particularly important:

- **FLOPS**: how much computation a GPU can perform per second.
- **FLOP/byte**: how much computation is performed for each byte of data moved.

These metrics describe different sides of the performance equation.

A GPU can have enormous peak FLOPS but still perform poorly on a particular workload if the workload cannot feed the compute units with data quickly enough.

This tutorial develops a practical mental model for understanding:

- FLOPS
- FLOP/byte / arithmetic intensity
- BF16, FP16, FP8, and INT8
- Tensor Core throughput
- Memory bandwidth
- The roofline model
- Large GEMM versus LLM inference
- Why quantization can improve inference performance

---

# 2. FLOPS: How Much Computation Can the GPU Do?

**FLOPS** means **floating-point operations per second**.

For example:

\[
1\ \text{TFLOPS} = 10^{12}\ \text{floating-point operations/second}
\]

and

\[
1\ \text{PFLOPS} = 10^{15}\ \text{floating-point operations/second}.
\]

Modern GPUs usually have different peak throughputs for different numerical formats.

A simplified example might look like:

| Datatype | Approximate relative compute throughput |
|---|---:|
| FP32 | 1× |
| BF16 / FP16 | much higher |
| FP8 | even higher |
| INT8 | potentially very high |

The exact numbers depend on the GPU architecture and whether operations use Tensor Cores / matrix engines.

The important point is:

> **The same mathematical operation can have very different peak throughput depending on the datatype.**

---

# 3. How Many FLOPs Does Matrix Multiplication Require?

Consider:

\[
C = AB
\]

where

\[
A\in\mathbb{R}^{M\times K},
\qquad
B\in\mathbb{R}^{K\times N},
\qquad
C\in\mathbb{R}^{M\times N}.
\]

Each element of \(C\) is computed as:

\[
C_{ij}
=
\sum_{k=1}^{K} A_{ik}B_{kj}.
\]

For each \(k\), we perform:

1. one multiplication
2. one addition

Therefore, approximately:

\[
2K
\]

floating-point operations are required per output element.

There are \(MN\) output elements, so the total work is:

\[
\boxed{
FLOPs = 2MKN
}
\]

This is the basic FLOP calculation for GEMM.

---

# 4. What Is FLOP/Byte?

FLOP/byte is also called **arithmetic intensity**.

It measures:

\[
\boxed{
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}{\text{Bytes moved}}
}
\]

It answers the question:

> How much computation do we get from each byte that needs to move through the memory system?

This is critical because GPUs have two major resources:

1. **Compute throughput**
2. **Memory bandwidth**

A workload can run out of either one first.

---

# 5. FLOP/Byte of a Simple GEMM

For

\[
C=AB
\]

we have approximately:

\[
FLOPs=2MKN.
\]

Suppose the matrices are read from memory and written back once.

If each element occupies \(s\) bytes, then:

\[
\text{Bytes}
\approx
(MK+KN+MN)s.
\]

Therefore:

\[
\boxed{
AI
\approx
\frac{2MKN}
{(MK+KN+MN)s}
}
\]

where \(AI\) is arithmetic intensity in FLOP/byte.

This equation immediately shows why datatype matters.

---

# 6. Why BF16 and FP8 Have Different Memory Requirements

The most basic difference is their element size.

| Datatype | Bits / element | Bytes / element |
|---|---:|---:|
| FP32 | 32 | 4 |
| BF16 | 16 | 2 |
| FP16 | 16 | 2 |
| FP8 | 8 | 1 |
| INT8 | 8 | 1 |
| INT4 | 4 | 0.5 |

For the same number of elements:

\[
\text{FP8 traffic}
\approx
\frac{1}{2}
\text{BF16 traffic}.
\]

Therefore, assuming the same computation and memory-access pattern:

\[
\boxed{
AI_{\mathrm{FP8}}
\approx
2AI_{\mathrm{BF16}}
}
\]

This is one of the fundamental advantages of lower-precision computation.

---

# 7. BF16 vs FP8: Two Things Change at Once

When moving from BF16 to FP8, two important things can change:

### 1. Memory traffic decreases

BF16:

\[
2\ \text{bytes/element}
\]

FP8:

\[
1\ \text{byte/element}
\]

So the same tensor requires approximately half the storage and memory bandwidth.

### 2. Tensor Core throughput increases

Modern GPUs often provide substantially higher Tensor Core throughput for FP8 than for BF16.

Conceptually:

```text
                 BF16                    FP8

element size      2 bytes                 1 byte
                    │                       │
                    ▼                       ▼
              more memory              less memory
                 traffic                  traffic
                    │                       │
                    ▼                       ▼
              lower AI                 higher AI
                                         (~2×)

              high compute            higher compute
               throughput              throughput
```

So FP8 can improve both sides of the equation:

- less data movement
- more peak computation

But this does **not** automatically mean every workload becomes 2× or 4× faster.

To understand why, we need the roofline model.

---

# 8. The Roofline Model

A useful first-order model for GPU performance is:

\[
\boxed{
P_{\mathrm{achieved}}
=
\min
\left(
P_{\mathrm{compute}},
BW_{\mathrm{memory}}\times AI
\right)
}
\]

where:

- \(P_{\mathrm{compute}}\) = peak compute throughput
- \(BW_{\mathrm{memory}}\) = memory bandwidth
- \(AI\) = arithmetic intensity

There are two possible bottlenecks.

## Compute-bound

If:

\[
P_{\mathrm{compute}}
<
BW_{\mathrm{memory}}\times AI
\]

then the workload is **compute-bound**.

The GPU's compute units are the limiting resource.

## Memory-bound

If:

\[
BW_{\mathrm{memory}}\times AI
<
P_{\mathrm{compute}}
\]

then the workload is **memory-bound**.

The GPU cannot supply data to the compute units quickly enough.

---

# 9. The Compute/Memory Crossover Point

The boundary between the two regimes occurs when:

\[
P_{\mathrm{compute}}
=
BW_{\mathrm{memory}}\times AI.
\]

Therefore:

\[
\boxed{
AI_{\mathrm{crossover}}
=
\frac{P_{\mathrm{compute}}}
{BW_{\mathrm{memory}}}
}
\]

This number tells us how much arithmetic intensity a workload needs to become compute-bound.

---

# 10. Example: BF16 vs FP8

Suppose, purely as an illustrative example, a GPU has:

```text
                 BF16       FP8
Peak compute     1 PFLOP    2 PFLOP
Memory BW        3 TB/s     3 TB/s
```

For BF16:

\[
AI_{\mathrm{crossover}}
=
\frac{1\ PFLOP}{3\ TB/s}
\approx333\ FLOP/byte.
\]

For FP8:

\[
AI_{\mathrm{crossover}}
=
\frac{2\ PFLOP}{3\ TB/s}
\approx667\ FLOP/byte.
\]

Notice something subtle:

> FP8 has higher compute throughput, so it actually requires **higher arithmetic intensity** before the workload becomes compute-bound.

This happens because the memory bandwidth has not doubled together with the compute throughput.

---

# 11. Why More FLOPS Can Make Memory Bottlenecks More Important

Suppose we increase Tensor Core performance:

```text
Before:

Compute      ███████████
Memory BW    ███████████


After:

Compute      ██████████████████████
Memory BW    ███████████
```

The compute engine becomes much faster, but the memory system did not necessarily become proportionally faster.

Therefore, workloads with insufficient arithmetic intensity can remain memory-bound.

This is why:

> **Peak FLOPS alone is not enough to predict application performance.**

You also need to understand the workload's FLOP/byte.

---

# 12. Large GEMM: Usually High Arithmetic Intensity

Consider a large matrix multiplication:

\[
[M,K]\times[K,N].
\]

When \(M\), \(N\), and \(K\) are large, each piece of data can be reused many times.

For example, a tile of matrix \(A\) can be loaded into shared memory and reused with many tiles of \(B\).

Conceptually:

```text
HBM
 │
 │ load data
 ▼
L2 / Shared Memory
 │
 │ reuse many times
 ▼
Tensor Cores
 │
 ├── multiply
 ├── multiply
 ├── multiply
 ├── multiply
 └── ...
```

This reuse greatly increases effective arithmetic intensity relative to HBM.

Large training GEMMs can therefore have extremely high FLOP/byte.

They are often **compute-bound**.

In this regime, higher FP8 Tensor Core throughput can translate directly into higher performance.

---

# 13. LLM Inference Is More Complicated

LLM inference contains several different computational regimes.

For example:

- prefill
- decode
- attention
- MLP
- KV-cache operations
- embedding lookup
- normalization
- elementwise operations

Their arithmetic intensities can be very different.

This is why saying:

> "The GPU has X PFLOPS of FP8"

does not tell us the actual performance of an LLM.

We need to know the workload shape and its arithmetic intensity.

---

# 14. Why Batch-1 Decode Can Be Memory-Bound

Consider a simplified matrix-vector operation:

\[
XW
\]

where:

\[
X=[1,d]
\]

and

\[
W=[d,d].
\]

Ignoring some details, the computation is approximately:

\[
2d^2\ FLOPs.
\]

If we need to read the entire weight matrix from memory:

### BF16

The weight matrix contains:

\[
d^2
\]

elements, each 2 bytes:

\[
2d^2\ bytes.
\]

Therefore:

\[
AI_{\mathrm{BF16}}
\approx
\frac{2d^2}{2d^2}
=
1\ FLOP/byte.
\]

### FP8

The weight matrix contains the same number of elements, but each requires only 1 byte:

\[
d^2\ bytes.
\]

Therefore:

\[
AI_{\mathrm{FP8}}
\approx
\frac{2d^2}{d^2}
=
2\ FLOP/byte.
\]

So in this simplified example:

\[
\boxed{
AI_{\mathrm{FP8}}
\approx
2AI_{\mathrm{BF16}}
}
\]

but both values are still very low compared with a large GEMM.

Therefore the operation is likely to be memory-bound.

---

# 15. Why Quantization Helps LLM Inference

Suppose a model has 100 billion parameters.

Ignoring metadata and other overhead:

## BF16

\[
100B\times2
=
200GB
\]

## FP8

\[
100B\times1
=
100GB
\]

## INT4

\[
100B\times0.5
=
50GB
\]

So:

| Format | Bytes / parameter | 100B parameters |
|---|---:|---:|
| BF16 | 2 | ~200 GB |
| FP8 | 1 | ~100 GB |
| INT4 | 0.5 | ~50 GB |

This has major implications for memory-bound inference.

If weight loading dominates execution time, reducing the number of bytes that must be read can have a large impact.

---

# 16. Why FP8 Can Improve Both Compute and Memory Efficiency

FP8 has two major effects:

```text
                    FP8
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
     1 byte/value         higher Tensor
          │               Core throughput
          ▼                     │
    less memory traffic         │
          │                     │
          └──────────┬──────────┘
                     ▼
              potentially
            higher performance
```

But the actual speedup depends on the bottleneck.

### If compute-bound

Higher FP8 Tensor Core throughput matters a lot.

### If memory-bound

Smaller FP8 representations reduce memory traffic.

### If neither dominates cleanly

The result is somewhere in between.

---

# 17. A Simple Performance Example

Suppose a workload has:

\[
AI=100\ FLOP/byte
\]

and the GPU has:

\[
BW=3\ TB/s.
\]

The memory-limited performance is:

\[
100\times3TB/s
=
300\ TFLOP/s.
\]

Suppose the GPU's peak compute is:

\[
1PFLOP/s.
\]

Then:

\[
P_{\mathrm{achieved}}
=
\min(1PFLOP/s,\ 300TFLOP/s)
\]

so:

\[
\boxed{
P_{\mathrm{achieved}}\approx300TFLOP/s
}
\]

The GPU is memory-bound.

Now suppose FP8 increases the arithmetic intensity to:

\[
200\ FLOP/byte.
\]

Then:

\[
200\times3TB/s
=
600TFLOP/s.
\]

If FP8 peak compute is 2 PFLOP/s:

\[
P_{\mathrm{achieved}}
=
\min(2PFLOP/s,\ 600TFLOP/s)
\]

giving:

\[
\boxed{
P_{\mathrm{achieved}}\approx600TFLOP/s
}
\]

The workload gets approximately 2× faster in this simplified model.

---

# 18. But Large GEMM Behaves Differently

Suppose instead the workload has:

\[
AI=1000\ FLOP/byte.
\]

For BF16:

\[
BW\times AI
=
3TB/s\times1000
=
3PFLOP/s.
\]

If BF16 compute peak is 1 PFLOP/s:

\[
P_{\mathrm{achieved}}
=
\min(1PFLOP/s,3PFLOP/s)
=
1PFLOP/s.
\]

It is compute-bound.

Now FP8 doubles the compute peak:

\[
P_{\mathrm{compute}}=2PFLOP/s.
\]

Even if the arithmetic intensity changes, the workload may remain compute-bound:

\[
P_{\mathrm{achieved}}
\approx2PFLOP/s.
\]

In this case, the higher FP8 compute throughput is the dominant benefit.

---

# 19. Arithmetic Intensity Depends on the Algorithm, Not Just the Datatype

It is tempting to say:

> "FP8 has twice the FLOP/byte of BF16."

That is only approximately true when comparing the same computation and memory-access pattern.

Actual arithmetic intensity depends on:

- matrix dimensions
- batch size
- sequence length
- tiling
- cache reuse
- shared-memory reuse
- fusion
- kernel implementation
- whether data is read from HBM or cache
- whether tensors are materialized or streamed
- communication between GPUs

So the real quantity should often be thought of as:

\[
AI_{\text{effective}}
=
\frac{\text{useful FLOPs}}
{\text{actual bytes transferred at a particular memory level}}.
\]

---

# 20. Memory Hierarchy Matters

A GPU does not have a single memory system.

A simplified hierarchy is:

```text
                    Registers
                       ▲
                       │
                 Shared Memory
                       ▲
                       │
                     L1/L2
                       ▲
                       │
                      HBM
```

The latency and bandwidth characteristics differ at every level.

A GEMM kernel may:

1. load weights from HBM
2. keep tiles in L2/shared memory
3. reuse them many times
4. perform many Tensor Core operations

Therefore, the same computation can have different effective arithmetic intensities depending on which memory level we are analyzing.

---

# 21. Why Tiling Is So Important

Consider matrix multiplication.

Without reuse:

```text
Load A → compute → discard
Load B → compute → discard
```

A lot of data movement is required.

With tiling:

```text
          HBM
           │
           ▼
       Load A tile
           │
           ▼
     Shared Memory
           │
     ┌─────┼─────┐
     ▼     ▼     ▼
   GEMM   GEMM   GEMM
     │     │     │
     └─────┼─────┘
           │
           ▼
       Reuse data
```

The same bytes participate in many FLOPs.

Therefore:

\[
\text{more reuse}
\Rightarrow
\text{higher FLOP/byte}
\Rightarrow
\text{less pressure on HBM}
\]

This is one of the fundamental principles behind high-performance GPU kernels.

---

# 22. FLOPS vs FLOP/Byte: The Key Difference

These two metrics answer different questions.

### FLOPS

> How fast can the hardware perform arithmetic?

### FLOP/byte

> How much arithmetic does the workload perform relative to its data movement?

For example:

```text
GPU capability:

                 FLOPS
                   │
                   ▼
             Tensor Cores
                   │
                   │
Workload ──────────┼──────────► Performance
                   │
                   ▲
              FLOP/byte
                   │
                   │
             Memory BW
```

Actual performance requires matching the workload's arithmetic intensity with the GPU's compute and memory capabilities.

---

# 23. The Most Useful Formula

For practical GPU performance analysis, remember:

\[
\boxed{
P_{\mathrm{achieved}}
=
\min
\left(
P_{\mathrm{peak}},
BW\times AI
\right)
}
\]

where:

\[
AI=
\frac{FLOPs}{Bytes}.
\]

And therefore:

\[
\boxed{
AI_{\mathrm{crossover}}
=
\frac{P_{\mathrm{peak}}}{BW}
}
\]

These two formulas are the foundation of the roofline model.

---

# 24. A Practical Workflow for GPU Performance Analysis

When analyzing a GPU workload, follow these steps.

## Step 1: Calculate the FLOPs

For GEMM:

\[
FLOPs\approx2MKN.
\]

For other operators, derive the operation count explicitly.

---

## Step 2: Calculate the data movement

Estimate:

- input bytes
- output bytes
- weight bytes
- intermediate tensors
- communication bytes

Do not automatically assume that every tensor is read exactly once. Kernel implementation and cache behavior matter.

---

## Step 3: Calculate arithmetic intensity

\[
AI=
\frac{FLOPs}{Bytes}.
\]

---

## Step 4: Find the GPU's compute peak

For the relevant datatype:

- BF16
- FP16
- FP8
- INT8
- FP32
- etc.

Use the Tensor Core / matrix-engine peak when appropriate.

---

## Step 5: Find the relevant memory bandwidth

For many workloads this may be HBM bandwidth, but for some kernels cache or shared-memory bandwidth can become important.

---

## Step 6: Apply the roofline model

Calculate:

\[
P_{\mathrm{memory}}
=
BW\times AI.
\]

Then compare:

\[
P_{\mathrm{memory}}
\]

with

\[
P_{\mathrm{compute}}.
\]

The smaller one gives the first-order performance ceiling.

---

# 25. Common Mistakes

## Mistake 1: Comparing GPUs only by peak FLOPS

A GPU with 2× the FLOPS does not necessarily run every workload 2× faster.

If the workload is memory-bound, increasing compute throughput may have little effect.

---

## Mistake 2: Assuming FP8 is always faster than BF16

FP8 can have higher peak throughput and lower memory traffic, but actual performance depends on:

- workload shape
- kernel support
- datatype conversion
- scaling
- numerical requirements
- memory bottlenecks
- communication
- hardware architecture

---

## Mistake 3: Ignoring batch size

Batch size can dramatically change arithmetic intensity.

Small batch:

```text
less computation
more weight traffic per operation
→ often memory-bound
```

Large batch:

```text
more computation
more reuse of weights
→ higher arithmetic intensity
→ potentially compute-bound
```

This is particularly important for LLM inference.

---

## Mistake 4: Treating model parameter size as execution time

A 100B-parameter model may contain 200 GB of BF16 weights, but that does not mean every inference must transfer exactly 200 GB from HBM.

Caching, tensor parallelism, weight residency, batching, and kernel implementation all matter.

---

# 26. Training vs Inference

A useful high-level distinction is:

| Workload | Typical arithmetic intensity | Typical bottleneck |
|---|---|---|
| Large training GEMM | Very high | Compute |
| Large-batch inference | High | Often compute |
| Prefill | Medium to high | Often compute / mixed |
| Batch-1 decode | Low | Often memory |
| KV-cache operations | Low | Often memory |
| Embedding lookup | Very low | Memory |
| Elementwise operations | Very low | Memory |

These are tendencies, not universal rules.

The actual bottleneck should be measured or estimated from the specific workload.

---

# 27. The Big Picture

The relationship between datatype, FLOPS, and memory can be summarized as:

```text
                     Datatype
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Bytes / element       Tensor Core throughput
              │                     │
              ▼                     ▼
       Memory traffic              FLOPS
              │                     │
              ▼                     │
         FLOP / byte                │
              │                     │
              └──────────┬──────────┘
                         ▼
                   Roofline model
                         │
                         ▼
               Actual performance
```

The important chain is:

\[
\boxed{
\text{Datatype}
\rightarrow
\text{Bytes/element}
\rightarrow
\text{Memory traffic}
\rightarrow
\text{FLOP/byte}
}
\]

and independently:

\[
\boxed{
\text{Datatype}
\rightarrow
\text{Tensor Core throughput}
\rightarrow
\text{Peak FLOPS}
}
\]

These two paths meet at the roofline:

\[
\boxed{
P_{\mathrm{achieved}}
=
\min
\left(
P_{\mathrm{peak}},
BW\times\frac{FLOPs}{Bytes}
\right)
}
\]

---

# 28. Final Mental Model

When looking at a GPU specification, don't stop at:

> **"This GPU has X PFLOPS of FP8."**

Instead ask three questions:

### 1. How much computation does my workload require?

\[
FLOPs
\]

### 2. How much data must I move?

\[
Bytes
\]

### 3. Which resource is the bottleneck?

\[
\boxed{
\min(P_{\mathrm{compute}}, BW\times FLOP/byte)
}
\]

That gives the most useful mental model for understanding GPU performance.

In particular:

- **BF16 → FP8** reduces bytes per element from 2 to 1.
- This can approximately double arithmetic intensity for the same memory-access pattern.
- FP8 also generally provides higher Tensor Core throughput on supported hardware.
- Large GEMMs can be compute-bound because they achieve high arithmetic intensity through data reuse.
- Small-batch LLM decode can be memory-bound because a large amount of model state is moved for relatively little computation.
- Quantization can therefore improve inference not only because of faster arithmetic, but also because it reduces memory traffic.
- Ultimately, **FLOPS tells you the compute capacity of the hardware, while FLOP/byte tells you how effectively your workload can use that compute capacity given the memory system.**

