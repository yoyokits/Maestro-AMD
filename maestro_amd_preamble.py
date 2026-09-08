"""Maestro AMD wrapper preamble — installed into env-amd's site-packages
by install_preamble.py, and executed at interpreter startup via the
companion maestro_amd_preamble.pth file.

**What this works around**

Maestro's models/wan/distributed/fsdp.py does an unconditional top-level
`from torch.distributed.fsdp import FullyShardedDataParallel as FSDP`.
`any2video.py` imports `shard_model` from that file but never calls it
anywhere on the single-GPU inference path — dead code. On any PyTorch
build where torch.distributed has no working backend (e.g. the AMD ROCm
nightly wheel for Windows, which ships without RCCL/GLOO), that import
chain crashes at `from torch._C._distributed_c10d import FakeProcessGroup`,
taking the whole app down for code nothing invokes.

Preemptively shadowing torch.distributed.fsdp in sys.modules *before* the
real module ever loads means Python's `from torch.distributed.fsdp import
...` finds our fake in the cache and skips running the real
fsdp/__init__.py entirely — the crashing import chain never executes.

**Why a .pth file and not sitecustomize.py**

This was originally shipped as `sitecustomize.py`. That works, but
site-packages has room for exactly **one** sitecustomize module: if any
dependency ever ships its own, one silently clobbers the other and the
FSDP crash returns with no diagnostic. Any number of `.pth` files
coexist, and a `.pth` line beginning with `import` is executed by site.py
during startup — slightly *earlier* than sitecustomize, and composably.
Same mechanism, same guarantees, no single-occupancy hazard.

**Why this shape and not others**

- Not a source patch to Maestro/app/models/wan/distributed/fsdp.py.
  A source patch works (was the original solution here — see git history
  for patch_fsdp.py) but is inherently invasive and has to be re-applied
  after every Update since `git reset --hard` wipes it. This file lives
  in the venv, which Update never touches — self-heals for free.
- Not a runtime probe. This file **must NEVER import torch at all** —
  not `torch`, not `torch.distributed`, and above all not
  `torch.distributed.fsdp`; nor call anything (like
  torch.distributed.is_available()) that could transitively trigger the
  real fsdp/__init__.py. On this ROCm-for-Windows nightly, importing
  torch from a startup hook spawns a runaway offload-arch.exe process
  storm (hundreds of processes in a loop): ROCm's `offload-arch` is
  itself a Python console script, so it re-runs this hook, which imports
  torch again, which spawns offload-arch again. Confirmed by direct
  observation twice, at wildly different severity. Writing to sys.modules
  unconditionally — never importing, never probing — avoids the storm
  entirely; verified live with a continuous-kill watchdog primed.
- Not `sys.modules['torch.distributed.fsdp'] = None`. That signals to
  Python's import machinery that the module doesn't exist and turns
  future `from torch.distributed.fsdp import X` into an ImportError,
  which Maestro doesn't handle. A fully-populated fake module with the
  attributes Maestro reads at def time (ShardingStrategy.FULL_SHARD, plus
  callable placeholders) satisfies the import cleanly.

**Trade-off:** on a healthy PyTorch build where torch.distributed.fsdp
works, this stub still wins the sys.modules race and shadows the real
module — so multi-GPU FSDP would break. That's acceptable here because
Maestro AMD is a single-GPU inference wrapper and shard_model() is never
invoked. If the upstream bug ever gets fixed and multi-GPU support is
wanted, delete this file, its .pth, and the install step in
install.js/update.js, and Maestro will use the real fsdp again.
"""
import sys
import types


def _install_fsdp_shadow():
    if "torch.distributed.fsdp" in sys.modules:
        return

    fsdp = types.ModuleType("torch.distributed.fsdp")
    fsdp.__path__ = []
    # Lets diagnose.py distinguish our stub from a real module without
    # importing anything.
    fsdp.__maestro_amd_stub__ = True
    wrap = types.ModuleType("torch.distributed.fsdp.wrap")
    wrap.__maestro_amd_stub__ = True

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "torch.distributed.fsdp is stubbed on this build — no working "
                "backend. (Maestro AMD wrapper preamble.)"
            )

    class ShardingStrategy:
        FULL_SHARD = "FULL_SHARD"
        SHARD_GRAD_OP = "SHARD_GRAD_OP"
        NO_SHARD = "NO_SHARD"
        HYBRID_SHARD = "HYBRID_SHARD"

    fsdp.FullyShardedDataParallel = _Unavailable
    fsdp.MixedPrecision = _Unavailable
    fsdp.ShardingStrategy = ShardingStrategy
    fsdp.wrap = wrap
    wrap.lambda_auto_wrap_policy = _Unavailable

    sys.modules["torch.distributed.fsdp"] = fsdp
    sys.modules["torch.distributed.fsdp.wrap"] = wrap


try:
    _install_fsdp_shadow()
except Exception:
    # A startup hook must never be able to break the interpreter.
    pass
