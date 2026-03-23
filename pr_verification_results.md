# PR Verification Results

Verified 2026-03-21 against available Docker images.

## Images Checked
- `rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260310`
- `rocm/sgl-dev:v0.5.9-rocm700-mi35x-20260319` (latest rocm700)
- `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260226`
- `rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260320` (latest rocm720)

All images show the same results -- none of the three PRs are fully merged.

## PR1: aiter MLA decode head-padding (aiter_backend.py)

**Status: NOT MERGED**

The code at line 227 checks `self.num_head == 16 or self.num_head == 128` to
disable the persistent MLA kernel for those specific head counts, but there is
**no logic to pad 8 query heads to 16** before calling `mla_decode_fwd`.

Kimi K2.5 at TP=8 has `num_head=8`. The PR would:
1. Pad 8 heads to 16 before `mla_decode_fwd`
2. Copy back valid 8 heads after
3. Disable persistent MLA kernel when `num_head % 16 != 0`

None of these changes are present in either image.

## PR2: aiter GEMM config tuning for small-batch decode

**Status: PARTIALLY PRESENT**

Found `gfx950-GEMM-A16W16-ATOMIC-N=256-K=7168.json` with `M_LEQ_16`, `M_LEQ_32`,
`M_LEQ_64`, `M_LEQ_128`, `M_LEQ_256` all using `BLOCK_SIZE_M=16`.

**Missing from the PR:**
- `M_LEQ_4` config (smallest batch specialization)
- Specialized config for N=384, K=7168 (router gate GEMM)
- MHA JIT sink/nsink variant fix

The generic `gfx950-GEMM-A16W16-ATOMIC.json` only has `"any"` with `BLOCK_SIZE_M=256`
(default, no small-batch optimization).

## PR3: sglang decode-path optimizations

**Status: NOT MERGED**

`rocm_linear_utils.py` (45 lines) uses `gemm_a16w16_atomic` for M<=256.
The PR would replace this with `F.linear` for M<=4 (185us -> 11us per MoE layer).

| Change | Expected | Found |
|--------|----------|-------|
| `F.linear` bypass for M<=4 router GEMM | Yes | No - uses `gemm_a16w16_atomic` for M<=256 |
| `BLOCK_SIZE_N=16, BLOCK_SIZE_K=256` for fused MoE M<=8 | Yes | No configs found |
| MSCCL++ pre-initialize fields | Yes | No hasattr/AttributeError guards found |
| Triton fallback for aiter varlen vision kernel | Yes | No fallback - only aiter path |
