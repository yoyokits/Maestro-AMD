"""Maestro AMD health check.

Prints everything needed to triage the failure modes documented in
CLAUDE.md "Known runtime issues", plus the AOTriton attention-backend
probe from PERFORMANCE.md section 0 -- the open question of whether this
particular GPU actually gets fast attention or silently falls back to the
O(n^2) math kernel that OOMs on real generation lengths.

Run from the Pinokio menu ("Diagnose"), or directly in the venv:

    python diagnose.py

Read-only and side-effect free: it installs nothing, changes nothing, and
is safe to run at any time, including while Maestro is running.

SAFETY -- do not "improve" this script by adding either of the following.
Both are documented crash/hang sources on this ROCm-for-Windows stack
(CLAUDE.md "Known runtime issues" #2):

  * `import torch.distributed.fsdp` (or anything under it)
  * `import torch.distributed.nn` (or anything under it)
  * `torch.distributed.is_available()`

Importing torch itself is fine here -- launch.py does it too. What is not
fine is doing it from sitecustomize.py, which runs in *every* interpreter
including the ones ROCm's own `offload-arch` console script spawns; that
feedback loop is what produced the runaway subprocess storm.
"""
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

REPO = Path(__file__).resolve().parent
APP = REPO / "Maestro" / "app"

OK = "  [ok]  "
WARN = "  [warn]"
BAD = "  [FAIL]"
INFO = "        "

_problems = []
_warnings = []


def section(title):
    print()
    print(f"=== {title} " + "=" * max(0, 60 - len(title)))


def fail(msg, hint=None):
    print(f"{BAD} {msg}")
    if hint:
        print(f"{INFO} -> {hint}")
    _problems.append(msg)


def warn(msg, hint=None):
    print(f"{WARN} {msg}")
    if hint:
        print(f"{INFO} -> {hint}")
    _warnings.append(msg)


def ok(msg):
    print(f"{OK} {msg}")


def info(msg):
    print(f"{INFO} {msg}")


# --------------------------------------------------------------------
def _wrapper_version():
    """Version of THIS wrapper, from its git tag.

    Distinct from the Maestro revision reported below, and from
    pinokio.js's `version` field (which is the Pinokio script schema
    version, not an app version -- see CHANGELOG.md).
    """
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO), "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown (not a git checkout?)"


def check_host():
    section("Host")
    info(f"wrapper       Maestro AMD {_wrapper_version()}")
    info(f"platform      {sys.platform} / {platform.machine()}")
    info(f"os            {platform.platform()}")
    info(f"python        {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        warn(
            f"Python {sys.version_info.major}.{sys.version_info.minor}, expected 3.11",
            "ROCm wheels do not ship cp310 builds; 3.11 is required.",
        )


def check_layout():
    section("Install layout")
    if not APP.exists():
        fail(
            f"missing {APP}",
            "Run Install from the Pinokio menu.",
        )
        return
    ok(f"upstream clone present ({APP.parent})")

    try:
        head = subprocess.run(
            ["git", "-C", str(APP.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=20,
        )
        if head.returncode == 0:
            info(f"Maestro HEAD  {head.stdout.strip()}")
        desc = subprocess.run(
            ["git", "-C", str(APP.parent), "log", "-1", "--format=%cd", "--date=short"],
            capture_output=True, text=True, timeout=20,
        )
        if desc.returncode == 0:
            info(f"Maestro date  {desc.stdout.strip()}")
    except Exception as exc:  # git missing / not a repo
        warn(f"could not read Maestro git revision: {exc}")

    markers = sorted((APP / "env-amd").glob(".maestro_amd_*.installed")) \
        if (APP / "env-amd").exists() else []
    if markers:
        ok(f"ROCm runtime marker: {markers[-1].name}")
    else:
        fail(
            "ROCm runtime marker missing",
            "torch.js never completed. Run Update (or Install) again.",
        )

    seedvc = APP / "postprocessing" / "seedvc" / "__init__.py"
    (ok if seedvc.exists() else warn)(
        f"seedvc component {'present' if seedvc.exists() else 'MISSING (Update self-heals it)'}"
    )

    ui_dist = APP.parent / "ui" / "dist"
    if ui_dist.exists():
        ok("React UI built")
    else:
        warn(
            "React UI not built -- only the Classic UI (/classic) will load",
            "Run Update to rebuild it.",
        )


def check_ffmpeg():
    section("ffmpeg / ffprobe  (CLAUDE.md #3, #4)")
    bin_dir = Path(sys.executable).parent
    ext = ".exe" if sys.platform == "win32" else ""
    for name in ("ffmpeg", "ffprobe"):
        found = shutil.which(name)
        if not found:
            # Distinguish "never provisioned" from "provisioned but this
            # shell didn't activate the venv" -- different problems with
            # different fixes. Pinokio's `venv:` param puts the venv's
            # Scripts/bin dir on PATH, so the second case is normal when
            # running diagnose.py by hand from an unactivated shell.
            local = bin_dir / f"{name}{ext}"
            if local.exists():
                warn(
                    f"{name} exists at {local} but is not on PATH",
                    "Normal if you ran this without activating the venv. "
                    "Maestro itself always runs with the venv active, so "
                    "this is only a problem if Maestro reports it too.",
                )
                found = str(local)
            else:
                fail(
                    f"{name} not installed",
                    "Run Update -- ensure_ffmpeg.py provisions both into the venv.",
                )
                continue
        try:
            r = subprocess.run(
                [found, "-version"], capture_output=True, text=True, timeout=20
            )
            first = r.stdout.splitlines()[0] if r.stdout else "(no version output)"
            ok(f"{name}: {first}")
        except Exception as exc:
            fail(f"{name} found at {found} but failed to run: {exc}")


def check_fsdp_shadow():
    section("FSDP import shadow  (CLAUDE.md #1)")
    # Deliberately checks sys.modules only. Never imports the real module.
    mod = sys.modules.get("torch.distributed.fsdp")
    if mod is None:
        warn(
            "torch.distributed.fsdp is not shadowed at startup",
            "Expected the wrapper preamble to pre-populate it. If Maestro "
            "crashes with ModuleNotFoundError: torch._C._distributed_c10d, "
            "run Update to reinstall the preamble.",
        )
    elif getattr(mod, "__maestro_amd_stub__", False):
        ok("torch.distributed.fsdp shadowed by the wrapper preamble")
    else:
        info("torch.distributed.fsdp already present (not our stub) -- "
             "this build may have a working distributed backend")

    dist_nn = sys.modules.get("torch.distributed.nn")
    if dist_nn is None:
        warn(
            "torch.distributed.nn is not shadowed at startup",
            "Expected the wrapper preamble to pre-populate it. If Maestro "
            "crashes with ImportError: cannot import name 'group' from "
            "'torch.distributed' (seen loading ace_step / "
            "vector_quantize_pytorch), run Update to reinstall the preamble.",
        )
    elif getattr(dist_nn, "__maestro_amd_stub__", False):
        ok("torch.distributed.nn shadowed by the wrapper preamble")
    else:
        info("torch.distributed.nn already present (not our stub) -- "
             "this build may have a working distributed backend")

    purelib = Path(sysconfig.get_paths()["purelib"])
    pth = purelib / "_maestro_amd_preamble.pth"
    legacy = purelib / "sitecustomize.py"
    if pth.exists():
        ok(f"preamble hook installed ({pth.name})")
    elif legacy.exists():
        info(f"legacy sitecustomize.py hook in use ({legacy})")
    else:
        fail(
            "no preamble hook found in site-packages",
            "Run Update to reinstall it.",
        )


def check_amd_defaults():
    """NVIDIA-specific defaults that must be overridden on AMD."""
    section("AMD setting overrides  (CLAUDE.md #6)")
    import json

    # MiniMax H3 text encoder: NVFP4 AWQ hangs the whole OS on AMD.
    defaults_dir = APP / "defaults"
    finetunes = APP / "finetunes"
    h3_types = sorted(p.stem for p in defaults_dir.glob("minimax_h3*.json"))         if defaults_dir.is_dir() else []
    if not h3_types:
        info("no MiniMax H3 models installed")
    else:
        missing, user_owned, seen = [], [], set()
        for mt in h3_types:
            f = finetunes / f"{mt}.json"
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                d = None
            enc = (d or {}).get("model", {}).get("minimax_h3_text_encoder_default")
            # Any variant except NVFP4 is safe. Which GGUF/int8 variant is
            # a speed/quality preference, not a correctness issue.
            if enc and enc != "nvfp4_awq":
                seen.add(enc)
                continue
            # A finetune the user wrote themselves is never overwritten by
            # the wrapper, so telling them to run Update would be wrong.
            if d is not None and not d.get("_maestro_amd_managed"):
                user_owned.append(mt)
            else:
                missing.append(mt)
        if missing:
            warn(
                f"H3 text-encoder override missing for {len(missing)}/{len(h3_types)} "
                "model(s): " + ", ".join(missing[:3]) + ("..." if len(missing) > 3 else ""),
                "Run Update -- NVFP4 AWQ is the stock default and hangs the "
                "whole OS on AMD.",
            )
        if user_owned:
            warn(
                "your own finetune overrides these H3 model(s) without setting "
                "a text encoder: " + ", ".join(user_owned),
                'Add "minimax_h3_text_encoder_default": "gguf_q4_k_m" to the '
                '"model" block of the matching file in Maestro/app/finetunes/. '
                "The wrapper will not edit a finetune you wrote.",
            )
        if not missing and not user_owned:
            which = ", ".join(sorted(seen)) or "(none)"
            ok(f"H3 text encoder default is {which} ({len(h3_types)} models)")
            if "nvfp4" not in which:
                info("If an H3 generation stalls, check whether a "
                     "'Fused 4-Step (Experimental)' variant is selected -- "
                     "see CLAUDE.md #6b.")

    # Any saved per-model setting still on NVFP4.
    stuck = []
    if (APP / "settings").is_dir():
        for f in sorted((APP / "settings").glob("minimax_h3*_settings.json")):
            try:
                if json.loads(f.read_text(encoding="utf-8")).get(
                        "minimax_h3_text_encoder") == "nvfp4_awq":
                    stuck.append(f.name)
            except Exception:
                pass
    if stuck:
        fail(
            "saved H3 settings still select NVFP4 AWQ: " + ", ".join(stuck),
            "This will hang your machine. Run Update to fix it.",
        )

    # Attention mode.
    try:
        cfg = json.loads((APP / "wgp_config.json").read_text(encoding="utf-8"))
        mode = cfg.get("attention_mode", "(unset)")
        bad = {"sage", "sage2", "sage3", "flash", "xformers", "radial", "sla"}
        if mode in bad:
            warn(
                f"attention_mode is {mode!r}, which is not installed on AMD",
                "Run Update -- it resets this to 'auto' (resolves to sdpa).",
            )
        else:
            ok(f"attention_mode = {mode!r}"
               + (" (resolves to sdpa on AMD)" if mode == "auto" else ""))
    except FileNotFoundError:
        info("wgp_config.json not created yet (first run)")
    except Exception as exc:
        warn(f"could not read wgp_config.json: {exc}")


def check_torch():
    section("PyTorch / ROCm")
    try:
        import torch
    except Exception as exc:
        fail(f"import torch failed: {exc}", "Run Update to reinstall the ROCm wheels.")
        return None

    info(f"torch         {torch.__version__}")
    hip = getattr(torch.version, "hip", None)
    cuda = getattr(torch.version, "cuda", None)
    if hip:
        ok(f"ROCm/HIP build (hip {hip})")
    elif cuda:
        fail(
            f"this is a CUDA build (cuda {cuda}), not ROCm",
            "requirements.txt pulled the PyPI CUDA wheel over the ROCm one. "
            "Delete the marker in Maestro/app/env-amd/ and run Update.",
        )
        return torch
    else:
        fail(
            "CPU-only torch build -- no GPU acceleration at all",
            "Delete the marker in Maestro/app/env-amd/ and run Update.",
        )
        return torch

    if not torch.cuda.is_available():
        fail(
            "torch.cuda.is_available() is False -- no GPU visible",
            "Check the AMD driver (Adrenalin 26.1.1+ for ROCm 7.2 on Windows).",
        )
        return torch

    try:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        gfx = getattr(props, "gcnArchName", "unknown").split(":")[0]
        total = props.total_memory / (1024 ** 3)
        ok(f"GPU: {name}")
        info(f"gfx target    {gfx}")
        info(f"VRAM          {total:.1f} GiB")
        return torch
    except Exception as exc:
        fail(f"could not query device 0: {exc}")
        return torch


def check_attention(torch):
    """The PERFORMANCE.md section 0 probe.

    Determines whether SDPA can actually reach a memory-efficient kernel
    on this GPU, or silently falls back to the math kernel that
    materializes the full O(n^2) attention matrix.
    """
    section("SDPA attention backends  (CLAUDE.md #5 / PERFORMANCE.md section 0)")
    if torch is None or not torch.cuda.is_available():
        info("skipped -- no usable GPU")
        return

    flag = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
    if flag == "1":
        ok("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 is set")
    else:
        warn(
            f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL is {flag!r}, expected '1'",
            "start.js sets this. Results below reflect the unset behaviour, "
            "not what Maestro actually runs with.",
        )

    try:
        import torch.nn.functional as F
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except Exception as exc:
        warn(f"could not import the SDPA backend selector: {exc}")
        return

    # Small enough to be instant, large enough that math vs flash differ.
    q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
    results = {}
    for label, backend in (
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("mem-efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ):
        try:
            with sdpa_kernel(backend):
                F.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            results[label] = True
            ok(f"{label:<14} available")
        except Exception as exc:
            results[label] = False
            first = str(exc).strip().splitlines()[0][:150]
            print(f"{WARN} {label:<14} unavailable: {first}")

    del q
    torch.cuda.empty_cache()

    if not results.get("flash") and not results.get("mem-efficient"):
        fail(
            "no fast attention backend on this GPU -- SDPA will use the math "
            "kernel, which allocates the full O(n^2) attention matrix",
            "Long or high-resolution generations will OOM (tens of GB "
            "requested). See PERFORMANCE.md section 0 -- AOTriton ships no "
            "kernels for RDNA 2, and gfx115x coverage is inconsistent. "
            "There is no env-var fix; keep generations short.",
        )
    else:
        ok("at least one memory-efficient attention backend is usable")


def check_triton():
    section("Triton  (PERFORMANCE.md section 3.1)")
    try:
        import triton
        ok(f"triton {triton.__version__} present -- torch.compile is reachable")
    except ImportError:
        if sys.platform.startswith("linux"):
            warn(
                "triton not installed",
                "Unexpected on Linux -- pytorch-triton-rocm normally arrives "
                "as a dependency of the ROCm torch wheel.",
            )
        else:
            info("not installed (expected on Windows -- AMD ships no ROCm "
                 "Triton wheel; see PERFORMANCE.md section 3.1)")
    except Exception as exc:
        warn(f"triton present but failed to import: {exc}")


def check_env():
    section("Runtime environment (as passed by start.js)")
    for key in (
        "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
        "PYTORCH_HIP_ALLOC_CONF",
        "HSA_ENABLE_SDMA",
        "MIOPEN_FIND_MODE",
        "MIOPEN_DISABLE_CACHE",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        val = os.environ.get(key)
        info(f"{key:<40} {val if val is not None else '(unset)'}")
    override = REPO / "user_env.json"
    if override.exists():
        ok(f"user env overrides active ({override.name})")
    else:
        info("user_env.json          (none -- see PERFORMANCE.md section 5)")


def main():
    print("Maestro AMD -- diagnostics")
    print("Attach this output when reporting an issue.")
    check_host()
    check_layout()
    check_env()
    check_ffmpeg()
    check_fsdp_shadow()
    check_amd_defaults()
    torch = check_torch()
    check_attention(torch)
    check_triton()

    section("Summary")
    if _problems:
        print(f"{BAD} {len(_problems)} problem(s):")
        for p in _problems:
            print(f"{INFO} - {p}")
    if _warnings:
        print(f"{WARN} {len(_warnings)} warning(s):")
        for w in _warnings:
            print(f"{INFO} - {w}")
    if not _problems and not _warnings:
        print(f"{OK} everything looks healthy.")
    print()
    # Always exit 0 -- this is a report, not a gate. A non-zero exit would
    # make Pinokio render a healthy diagnostic run as a failed script.
    return 0


if __name__ == "__main__":
    sys.exit(main())
