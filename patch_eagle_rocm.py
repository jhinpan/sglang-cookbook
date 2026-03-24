"""Patch SGLang to enable non-greedy Eagle3 speculative decoding on ROCm/HIP.

Creates a PyTorch fallback module for 3 missing sgl_kernel C++ ops
(top_k_renorm_prob, top_p_renorm_prob, tree_speculative_sampling_target_only)
and patches the SGLang source to use them when running on HIP.

Changes:
1. Installs rocm_kernel_fallbacks.py into the speculative module
2. spec_utils.py: TREE_SPEC_KERNEL_AVAILABLE = _is_cuda or _is_hip
3. eagle_info.py: import fallbacks on HIP instead of sgl_kernel
4. eagle_info_v2.py: import fallbacks on HIP, fix greedy-only guard

Usage:
    docker cp sglang-cookbook/rocm_kernel_fallbacks.py CONTAINER:/tmp/
    docker cp sglang-cookbook/patch_eagle_rocm.py CONTAINER:/tmp/
    docker exec CONTAINER python /tmp/patch_eagle_rocm.py
"""

import pathlib
import shutil
import sys

SGLANG_ROOT = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt")
SCRIPT_DIR = pathlib.Path(__file__).parent

_ok = 0
_skip = 0
_fail = 0


def _report(path, action, success):
    global _ok, _skip, _fail
    tag = "OK" if success else "SKIP"
    if success:
        _ok += 1
    else:
        _skip += 1
    print(f"  [{tag}] {path.name}: {action}")


def _patch(path, old, new, label):
    content = path.read_text()
    if old in content:
        content = content.replace(old, new, 1)
        path.write_text(content)
        _report(path, label, True)
        return True
    else:
        _report(path, f"{label} -- pattern not found (already patched?)", False)
        return False


# ── 1. Install the fallback module ──────────────────────────────────────────

def install_fallback_module():
    dst = SGLANG_ROOT / "speculative" / "rocm_kernel_fallbacks.py"
    src = SCRIPT_DIR / "rocm_kernel_fallbacks.py"
    if not src.exists():
        print(f"ERROR: {src} not found. Copy it alongside this script.")
        sys.exit(1)
    shutil.copy2(src, dst)
    print(f"  [OK] Installed {dst}")


# ── 2. Patch spec_utils.py ──────────────────────────────────────────────────

def patch_spec_utils():
    p = SGLANG_ROOT / "speculative" / "spec_utils.py"
    _patch(
        p,
        "TREE_SPEC_KERNEL_AVAILABLE = _is_cuda  # This kernel is only available for CUDA now",
        "TREE_SPEC_KERNEL_AVAILABLE = _is_cuda or _is_hip",
        "TREE_SPEC_KERNEL_AVAILABLE = _is_cuda or _is_hip",
    )


# ── 3. Patch eagle_info.py ──────────────────────────────────────────────────

def patch_eagle_info():
    p = SGLANG_ROOT / "speculative" / "eagle_info.py"

    # 3a. Add is_hip to the utils import
    _patch(
        p,
        "from sglang.srt.utils import is_cuda, next_power_of_2",
        "from sglang.srt.utils import is_cuda, is_hip, next_power_of_2",
        "added is_hip to utils import",
    )

    # 3b. Add elif is_hip() branch for the sgl_kernel imports
    OLD_IMPORT = """\
if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )"""

    NEW_IMPORT = """\
if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )
elif is_hip():
    from sglang.srt.speculative.rocm_kernel_fallbacks import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )"""

    _patch(p, OLD_IMPORT, NEW_IMPORT, "added elif is_hip() import branch")


# ── 4. Patch eagle_info_v2.py ───────────────────────────────────────────────

def patch_eagle_info_v2():
    p = SGLANG_ROOT / "speculative" / "eagle_info_v2.py"

    # 4a. Add elif _is_hip branch for the sgl_kernel imports
    OLD_IMPORT_V2 = """\
if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )"""

    NEW_IMPORT_V2 = """\
if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )
elif _is_hip:
    from sglang.srt.speculative.rocm_kernel_fallbacks import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )"""

    _patch(p, OLD_IMPORT_V2, NEW_IMPORT_V2, "added elif _is_hip import branch")

    # 4b. Add TREE_SPEC_KERNEL_AVAILABLE to the spec_utils import
    _patch(
        p,
        """\
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)""",
        """\
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    TREE_SPEC_KERNEL_AVAILABLE,
    generate_simulated_accept_index,
)""",
        "added TREE_SPEC_KERNEL_AVAILABLE to spec_utils import",
    )

    # 4c. Fix the greedy guard in sample() -- replace hardcoded _is_hip with flag
    _patch(
        p,
        "if sampling_info.is_all_greedy or _is_npu or _is_hip:",
        "if sampling_info.is_all_greedy or _is_npu or not TREE_SPEC_KERNEL_AVAILABLE:",
        "replaced _is_hip greedy guard with TREE_SPEC_KERNEL_AVAILABLE flag",
    )


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Eagle3 ROCm non-greedy patch")
    print("=" * 50)

    print("\n1. Installing fallback module...")
    install_fallback_module()

    print("\n2. Patching spec_utils.py...")
    patch_spec_utils()

    print("\n3. Patching eagle_info.py...")
    patch_eagle_info()

    print("\n4. Patching eagle_info_v2.py...")
    patch_eagle_info_v2()

    print(f"\nDone: {_ok} applied, {_skip} skipped")
    if _skip > 0:
        print("(Skipped patches may already be applied from a previous run)")
    print("\nRestart the SGLang server to take effect.")
