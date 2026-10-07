# GPU Kernels and Attention

[Back to portfolio](../README.md)

## [Annotated FlashAttention](01_annotated_flash_attention.ipynb)

PyTorch reference code exploring online softmax and blockwise attention forward and backward passes.

## [Triton Vector Addition](02_vector_addition.ipynb)

Elementwise and blocked vector addition, launch grids and timing examples.

## [Triton Naive Matrix Multiplication](03_naive_matmul.ipynb)

Scalar dot-product pseudocode and a basic Triton matrix multiplication kernel.

## [Triton Blocked Matrix Multiplication](04_blocked_matmul.ipynb)

Tiled matrix multiplication and autotuning configuration code.

## [Triton Grouped Matrix Multiplication](05_grouped_matmul.ipynb)

Grouped tile ordering, autotuning, correctness-test code and benchmark scaffolding.

## [Triton Softmax](06_softmax.ipynb)

Row-wise softmax forward and backward kernels and DLPack wrappers.

## [Triton LayerNorm](07_layernorm.ipynb)

Layer normalization forward and backward kernel implementations.

## [Triton Embedding and Atomic Sum](08_embedding_atomic_sum.ipynb)

Embedding lookup and backward accumulation with atomic addition.

## [Triton FlashAttention](09_flash_attention.ipynb)

Attention forward and backward kernels, causal masking and grouped-query head handling.

## Environment and status

Requires a supported NVIDIA GPU, CUDA-compatible PyTorch and Triton. Some notebooks also import CuPy; choose a build matching your CUDA environment. Grouped matmul contains pytest-based test scaffolding. Benchmark code is included, but no speedup or numerical-correctness claim has been verified for this import.

Open the notebook in JupyterLab or a suitable GPU notebook environment and inspect its configuration before execution. These are learning implementations; some cells are incomplete or require corrections. See [validation notes](../VALIDATION.md).

Original upload titles are recorded in [SOURCES.md](../SOURCES.md). Exact tutorial/repository attribution and the distinction between reproduced code and personal extensions remain to be documented by the author.

## [FlashAttention with Autograd](10_flash_attention_autograd.ipynb)

Triton forward/backward attention kernels, a custom autograd wrapper and comparison-test code.

Requires a compatible NVIDIA GPU, PyTorch and Triton. Contains a Python parsing error and a `float["-inf"]` expression in reference-test code. Numerical comparisons have not been executed.
