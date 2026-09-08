"""Assert that the installed torch is the ROCm build, not a CPU or CUDA one.

Two modes:

  (no args)     Post-install check. Writes .torch_needs_reinstall next to
                the venv when the build is wrong; install.js / update.js /
                repair.js gate a torch.js re-run on that flag. Always
                exits 0 -- the flag carries the signal.

  --preflight   Pre-launch check, run by start.js. Writes .torch_broken
                when the build is *certainly* wrong, so start.js can show
                a readable explanation instead of letting launch.py die
                ~40 lines deep with "Torch not compiled with CUDA
                enabled". Deliberately metadata-only and conservative:
                it never imports torch (startup is not the place to pay
                that) and never flags an unrecognised version string.

**Why this exists.** Nothing in Maestro's requirements.txt pins torch,
but torchcodec / accelerate / peft / timm / open_clip_torch all depend on
it. If the resolver ever decides the installed ROCm wheel doesn't satisfy
a constraint, it silently installs a stock PyPI torch over the top --
which on Windows is CPU-only (`2.14.0+cpu`). Everything then imports
fine, prints `CUDA unavailable`, and crashes much later inside
`models/wan/modules/t5.py` at `torch.cuda.current_device()` with a
message that says nothing about the real cause.

That is a real, reported user failure, not a hypothetical:

    [Runtime] PyTorch 2.14.0+cpu | CUDA none | CUDA unavailable (unknown)
    ...
    AssertionError: Torch not compiled with CUDA enabled

It is especially easy to reach via Update, because update.js skips
torch.js whenever the runtime marker exists -- so a requirements pass
could replace torch and nothing would put it back.
"""
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REINSTALL_FLAG = ".torch_needs_reinstall"
BROKEN_FLAG = ".torch_broken"


def _venv_root():
    # sysconfig purelib is <venv>/Lib/site-packages (Windows) or
    # <venv>/lib/pythonX.Y/site-packages (POSIX); walk up to the venv root.
    root = Path(sysconfig.get_paths()["purelib"])
    for _ in range(4):
        if (root / "pyvenv.cfg").exists():
            break
        root = root.parent
    return root


def _certainly_wrong(raw):
    """Metadata-only verdict. Returns a reason string, or None if fine.

    Only reports what the version string proves. An unrecognised string
    (AMD's Windows nightlies don't always carry a +rocm local tag) is
    treated as acceptable -- a false positive here would block startup
    for a working install, which is far worse than a late failure.
    """
    low = raw.lower()
    if "rocm" in low or "hip" in low:
        return None
    if low.endswith("+cpu") or "+cpu" in low:
        return (
            f"torch {raw} is a CPU-only build. Maestro needs the AMD ROCm "
            "build; generation cannot run on this one."
        )
    if "+cu" in low or "cuda" in low:
        return (
            f"torch {raw} is an NVIDIA CUDA build. Maestro AMD needs the "
            "ROCm build; this one cannot see an AMD GPU."
        )
    return None


def _verdict():
    """Full verdict for the post-install check. May import torch."""
    try:
        raw = version("torch")
    except PackageNotFoundError:
        return False, "torch is not installed"

    reason = _certainly_wrong(raw)
    if reason:
        return False, reason
    low = raw.lower()
    if "rocm" in low or "hip" in low:
        return True, f"torch {raw}"

    # Ambiguous -- fall back to the authoritative check.
    try:
        import torch
    except Exception as exc:
        return False, f"torch {raw} (import failed: {exc})"
    if getattr(torch.version, "hip", None):
        return True, f"torch {raw} (hip {torch.version.hip})"
    if getattr(torch.version, "cuda", None):
        return False, f"torch {raw} (cuda {torch.version.cuda})"
    return False, f"torch {raw} (CPU-only build)"


def _set(path, text):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except OSError as exc:
        print(f"[verify_rocm_torch] could not write {path}: {exc}")
        return False


def _clear(path):
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def preflight():
    """Fast, conservative, no torch import. Used by start.js."""
    flag = _venv_root() / BROKEN_FLAG
    try:
        raw = version("torch")
    except PackageNotFoundError:
        _set(flag, "torch is not installed\n")
        print("[verify_rocm_torch] torch is not installed")
        return 0

    reason = _certainly_wrong(raw)
    if reason:
        print(f"[verify_rocm_torch] {reason}")
        _set(flag, reason + "\n")
    else:
        _clear(flag)
    return 0


def main(argv):
    if "--preflight" in argv:
        return preflight()

    flag = _venv_root() / REINSTALL_FLAG
    is_rocm, detail = _verdict()

    if is_rocm:
        print(f"[verify_rocm_torch] ok: {detail}")
        _clear(flag)
        _clear(_venv_root() / BROKEN_FLAG)
        return 0

    print(f"[verify_rocm_torch] WRONG BUILD: {detail}")
    print("[verify_rocm_torch] the dependency resolver replaced the ROCm "
          "wheels; scheduling a torch reinstall")
    if not _set(flag, f"{detail}\n"):
        # Without the flag the caller can't self-heal, so make it loud.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
