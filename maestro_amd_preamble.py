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

Same failure family, different call site: `vector_quantize_pytorch`'s
`lookup_free_quantization.py` (pulled in by the ace_step TTS pipeline) does
`from torch.distributed import nn as dist_nn` at module scope. That triggers
PyTorch's real `torch/distributed/nn/__init__.py`, which itself does
`from torch.distributed import group` — undefined when
`torch.distributed.is_available()` is False, so the import dies with
`ImportError: cannot import name 'group' from 'torch.distributed'`. The one
thing `dist_nn` is used for (`dist_nn.all_reduce(...)` in
`lookup_free_quantization.py`) is itself guarded by a world-size-greater-
than-1 check, so it's dead code on single-GPU inference — same shape as
`shard_model()`. Shadowed below the same way.

**Opt-in escape hatch: audio-separator on CPU.** `audio-separator` (the
RoFormer vocal extractor Maestro runs on Director song uploads) picks its
device with `torch.cuda.is_available()`, which is True on ROCm, and ignores
the caller's `torch.set_default_device('cpu')`. If a GPU cannot run that
model, the crash takes the whole server down with it. Setting
`MAESTRO_AMD_AUDIO_SEPARATOR_DEVICE=cpu` (via user_env.json) installs a
post-import hook that wraps `Separator.__init__` to move new instances to
CPU. **Not on by default:** on a working GPU it is drastically slower —
measured on an RX 7900 XTX, a 60 s clip took 19 s on the GPU and had not
finished one of eight chunks after 11 minutes on twelve CPU threads. See
CLAUDE.md #13.

This one cannot be a sys.modules shadow — the real module must load, and
only one method needs replacing afterwards — so it is a `sys.meta_path`
post-import hook instead. Registering it imports nothing; torch is only
read out of sys.modules after audio_separator has imported it itself, long
after startup.

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
import functools
import os
import sys
import types

AUDIO_SEPARATOR_PACKAGE = "audio_separator.separator"
AUDIO_SEPARATOR_DEVICE_ENV = "MAESTRO_AMD_AUDIO_SEPARATOR_DEVICE"
HOOK_MARKER = "__maestro_amd_hook__"
PATCH_MARKER = "__maestro_amd_cpu__"


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


def _install_distributed_nn_shadow():
    if "torch.distributed.nn" in sys.modules:
        return

    dist_nn = types.ModuleType("torch.distributed.nn")
    dist_nn.__path__ = []
    dist_nn.__maestro_amd_stub__ = True
    functional = types.ModuleType("torch.distributed.nn.functional")
    functional.__maestro_amd_stub__ = True

    class _Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "torch.distributed.nn is stubbed on this build — no working "
                "backend. (Maestro AMD wrapper preamble.)"
            )

    for name in (
        "all_reduce",
        "all_gather",
        "all_to_all",
        "broadcast",
        "gather",
        "scatter",
        "reduce_scatter",
    ):
        setattr(dist_nn, name, _Unavailable)
        setattr(functional, name, _Unavailable)

    dist_nn.functional = functional
    sys.modules["torch.distributed.nn"] = dist_nn
    sys.modules["torch.distributed.nn.functional"] = functional


def separator_cpu_requested():
    return os.environ.get(AUDIO_SEPARATOR_DEVICE_ENV, "").strip().lower() == "cpu"


def force_separator_cpu(module):
    """Wrap module.Separator.__init__ so new instances run on CPU.

    Only the public class and the two attributes its load_model() hands to
    the model are relied on -- stable from audio-separator 0.36 to 0.47,
    unlike the internal setup_torch_device().
    """
    separator = getattr(module, "Separator", None)
    if not isinstance(separator, type) or separator.__dict__.get(PATCH_MARKER):
        return
    original_init = separator.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            device = getattr(self, "torch_device", None)
            if device is None or getattr(device, "type", "cpu") == "cpu":
                return
            torch = sys.modules.get("torch")
            if torch is None:
                return
            self.torch_device = torch.device("cpu")
            self.onnx_execution_provider = ["CPUExecutionProvider"]
            msg = (
                "Maestro AMD: running audio-separator on CPU -- much slower, "
                f"requested via {AUDIO_SEPARATOR_DEVICE_ENV}=cpu"
            )
            logger = getattr(self, "logger", None)
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)
        except Exception:
            pass

    separator.__init__ = __init__
    setattr(separator, PATCH_MARKER, True)


class _PatchingLoader:
    def __init__(self, loader, patch):
        self._loader = loader
        self._patch = patch

    def exec_module(self, module):
        # Hand the module back to the real loader before running it, so
        # nothing downstream ever sees this wrapper.
        try:
            module.__loader__ = self._loader
            module.__spec__.loader = self._loader
        except Exception:
            pass
        self._loader.exec_module(module)
        try:
            self._patch(module)
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _PostImportHook:
    """sys.meta_path entry that patches one module right after it loads."""

    def __init__(self, fullname, patch):
        self.fullname = fullname
        self._patch = patch
        setattr(self, HOOK_MARKER, fullname)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.fullname:
            return None
        spec = None
        for finder in list(sys.meta_path):
            if finder is self or getattr(finder, HOOK_MARKER, None) == fullname:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                break
        if spec is None or not hasattr(spec.loader, "exec_module"):
            return spec
        spec.loader = _PatchingLoader(spec.loader, self._patch)
        return spec


def _install_audio_separator_cpu_hook():
    if not separator_cpu_requested():
        return
    if AUDIO_SEPARATOR_PACKAGE in sys.modules:
        force_separator_cpu(sys.modules[AUDIO_SEPARATOR_PACKAGE])
        return
    for finder in sys.meta_path:
        if getattr(finder, HOOK_MARKER, None) == AUDIO_SEPARATOR_PACKAGE:
            return
    sys.meta_path.insert(0, _PostImportHook(AUDIO_SEPARATOR_PACKAGE, force_separator_cpu))


try:
    _install_fsdp_shadow()
except Exception:
    # A startup hook must never be able to break the interpreter.
    pass

try:
    _install_distributed_nn_shadow()
except Exception:
    # A startup hook must never be able to break the interpreter.
    pass

try:
    _install_audio_separator_cpu_hook()
except Exception:
    # A startup hook must never be able to break the interpreter.
    pass
