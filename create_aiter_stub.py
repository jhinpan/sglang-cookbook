"""Create a minimal aiter stub package that satisfies all imports without the C++ extension."""
import pathlib, shutil, enum

aiter_dir = pathlib.Path("/sgl-workspace/aiter_stub/aiter")

# Remove old stub if exists
if aiter_dir.parent.exists():
    shutil.rmtree(aiter_dir.parent)
aiter_dir.mkdir(parents=True)

# Root __init__.py with dummy classes
(aiter_dir / "__init__.py").write_text('''
import enum as _enum

class ActivationType(_enum.IntEnum):
    Identity = 0
    Gelu = 1
    GeluTanh = 2
    Relu = 3
    Silu = 4

class QuantType(_enum.IntEnum):
    No = 0
    per_Tensor = 1
    per_Token = 2
    per_1x128 = 3
    per_128x128 = 4

class dtypes:
    fp8 = "fp8_e4m3fnuz"

def gemm_a8w8_bpreshuffle(*a, **kw): raise NotImplementedError("aiter stub")
def get_hip_quant(*a, **kw): raise NotImplementedError("aiter stub")
''')

# fused_moe submodule
(aiter_dir / "fused_moe.py").write_text('''
def fused_moe(*a, **kw): raise NotImplementedError("aiter stub")
''')

# ops package
ops = aiter_dir / "ops"
ops.mkdir()
(ops / "__init__.py").write_text("")

# ops.shuffle
shuffle = ops / "shuffle.py"
shuffle.write_text('''
def shuffle_weight(*a, **kw): raise NotImplementedError("aiter stub")
def shuffle_weight_fp4(*a, **kw): raise NotImplementedError("aiter stub")
''')

# ops.triton package
triton = ops / "triton"
triton.mkdir()
(triton / "__init__.py").write_text("")

# ops.triton.quant
quant = triton / "quant"
quant.mkdir()
(quant / "__init__.py").write_text('''
def dynamic_mxfp4_quant(*a, **kw): raise NotImplementedError("aiter stub")
def fused_fp8_quant(*a, **kw): raise NotImplementedError("aiter stub")
''')
(quant / "fused_fp8_quant.py").write_text('''
def fused_fp8_quant(*a, **kw): raise NotImplementedError("aiter stub")
''')

# ops.triton.gemm subpackages
gemm = triton / "gemm_a8w8_blockscale.py"
gemm.write_text('def gemm_a8w8_blockscale(*a, **kw): raise NotImplementedError("aiter stub")')

(triton / "gemm_afp4wfp4.py").write_text('def gemm_afp4wfp4(*a, **kw): raise NotImplementedError("aiter stub")')
(triton / "gemm_afp4wfp4_pre_quant_atomic.py").write_text('def gemm_afp4wfp4_pre_quant(*a, **kw): raise NotImplementedError("aiter stub")')
(triton / "fused_mxfp4_quant.py").write_text('def fused_mxfp4_quant(*a, **kw): raise NotImplementedError("aiter stub")')
(triton / "batched_gemm_afp4wfp4_pre_quant.py").write_text('def batched_gemm_afp4wfp4_pre_quant(*a, **kw): raise NotImplementedError("aiter stub")')

# ops.triton.gemm.fused
fused = triton / "gemm"
fused.mkdir()
(fused / "__init__.py").write_text("")
fused2 = fused / "fused"
fused2.mkdir()
(fused2 / "__init__.py").write_text("")
(fused2 / "fused_gemm_afp4wfp4_split_cat.py").write_text('def fused_gemm_afp4wfp4_split_cat(*a, **kw): raise NotImplementedError("aiter stub")')

# utility package
util = aiter_dir / "utility"
util.mkdir()
(util / "__init__.py").write_text("")
(util / "dtypes.py").write_text('fp8 = "fp8_e4m3fnuz"')
(util / "fp4_utils.py").write_text('def e8m0_shuffle(*a, **kw): raise NotImplementedError("aiter stub")')

print("Created aiter stub at", aiter_dir.parent)
