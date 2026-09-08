"""Install this wrapper's startup preamble into the active venv.

Copies maestro_amd_preamble.py into the venv's site-packages and writes a
companion .pth file whose single `import` line CPython executes at
interpreter startup, before any user code runs.

See maestro_amd_preamble.py for full context on what it works around and
why this shape (as opposed to a source patch, a runtime probe, an
sys.modules=None sentinel, or the sitecustomize.py this replaces).

Safe to re-run: overwrites in place, no state to keep.
"""
import shutil
import sys
import sysconfig
from pathlib import Path

MODULE = "maestro_amd_preamble"
# Leading underscore so it sorts early among .pth files -- site.py
# processes them in directory-listing order, and shadowing
# torch.distributed.fsdp before anything else touches torch is the whole
# point.
PTH_NAME = "_maestro_amd_preamble.pth"
LEGACY_MARKER = "Maestro AMD wrapper preamble"


def main():
    src = Path(__file__).parent / f"{MODULE}.py"
    if not src.exists():
        print(f"[install_preamble] source missing: {src}")
        sys.exit(1)

    # sysconfig.get_paths()['purelib'] is the venv's own site-packages
    # (the canonical spot for pure-Python packages) -- more reliable than
    # scanning site.getsitepackages(), which on Windows venvs also lists
    # the venv root and the base interpreter's site-packages.
    dest_dir = Path(sysconfig.get_paths()["purelib"])
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / f"{MODULE}.py"
    shutil.copy2(src, dest)
    print(f"[install_preamble] {src} -> {dest}")

    pth = dest_dir / PTH_NAME
    # A .pth line starting with "import" is executed by site.py at
    # startup. Everything else in a .pth is treated as a path entry.
    pth.write_text(f"import {MODULE}\n", encoding="utf-8")
    print(f"[install_preamble] wrote {pth}")

    # Retire the previous sitecustomize.py hook, but only if it is ours --
    # site-packages has room for exactly one, and clobbering someone
    # else's would be precisely the bug this migration exists to avoid.
    legacy = dest_dir / "sitecustomize.py"
    if legacy.exists():
        try:
            if LEGACY_MARKER in legacy.read_text(encoding="utf-8", errors="replace"):
                legacy.unlink()
                print(f"[install_preamble] removed superseded {legacy}")
            else:
                print(
                    f"[install_preamble] leaving third-party {legacy} alone "
                    "(not ours)"
                )
        except OSError as exc:
            print(f"[install_preamble] could not inspect {legacy}: {exc}")


if __name__ == "__main__":
    main()
