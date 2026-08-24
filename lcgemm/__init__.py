"""Low-complexity GEMM for Blackwell, with the pre-GEMM seam fused in.

A rank-R bilinear decomposition replaces the p*q*s naive block products with R,
so the GEMM consumes R linear functionals ``L_r(X)`` of the activation matrix
rather than ``X``.  And ``X`` is never what is in HBM -- it is ``f(Y)`` for some
cheap row-local or pointwise ``f`` (an RMSNorm, a SwiGLU, a sigmoid gate) left by
the preceding kernel.  So one kernel reads ``Y`` and emits the R A-planes, and
``X`` never lands.

Layout
------

``scheme``        the decomposition: which R products, and how C is rebuilt
``planes``        the offline B-plane preparation and fp32 references
``chain_gauge``   the interleaved-K gauge the chained epilogue emits in
``dlpack``        memoised tensor wrapping for the compiled CuTe kernels
``kernels/``      the CuTe DSL kernels -- GEMMs and fused A-plane producers
``seams/``        one ``Plan`` per fusable seam: what the model actually calls
``integrate/``    the vLLM swap: custom ops, the layer forward, arm/install
``bench/``        measurement, validation, and headline aggregation

Nothing under ``integrate/`` imports anything under ``bench/``; the deployed
path never pulls in the benchmark harness.
"""

__version__ = "0.1.0"
