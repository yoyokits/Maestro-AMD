"""Update bookkeeping for the Maestro AMD wrapper.

Three jobs, selected by subcommand, all operating on the upstream clone
in ./Maestro:

  record      Save the current upstream HEAD before `git reset --hard`
              so "Roll back last update" has somewhere to go. Without
              this, a bad upstream commit leaves Reset -- which deletes
              every downloaded model -- as the only escape.

  ui-check    Decide whether the React UI needs rebuilding. `npm install
              && npm run build` runs on every Update today, including the
              everyday "Update & Start" path, even when nothing under
              ui/ changed. Writes .maestro_state/ui_rebuild when a rebuild
              is actually needed; update.js gates the npm steps on it.

  ui-done     Record the HEAD the current UI bundle was built from and
              clear the rebuild flag.

Every subcommand is best-effort: if git is unavailable or the repo is in
an unexpected state, it fails *open* (schedules the rebuild, skips the
rollback point) rather than blocking an update.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
MAESTRO = REPO / "Maestro"
UI = MAESTRO / "ui"

# Wrapper state lives OUTSIDE the upstream clone. Two reasons: it keeps
# `git status` in Maestro/ clean, and -- the one that actually bit --
# repair.js runs `git clean -fdx` inside Maestro/, which would otherwise
# delete the rollback point at exactly the moment a user needs it most.
STATE = REPO / ".maestro_state"
PREV_HEAD = STATE / "prev_head"
UI_BUILT = STATE / "ui_built"
UI_REBUILD = STATE / "ui_rebuild"


def _git(*args, check=False):
    return subprocess.run(
        ["git", "-C", str(MAESTRO), *args],
        capture_output=True, text=True, timeout=60, check=check,
    )


def _head():
    r = _git("rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def _write(path, text):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return True
    except OSError as exc:
        print(f"[update_state] could not write {path}: {exc}")
        return False


def _clear(path):
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def cmd_record():
    if not MAESTRO.exists():
        print("[update_state] no Maestro clone yet -- nothing to record")
        return 0
    head = _head()
    if not head:
        print("[update_state] could not read HEAD -- rollback point not saved")
        return 0
    if _write(PREV_HEAD, head + "\n"):
        print(f"[update_state] rollback point saved: {head[:12]}")
    return 0


def cmd_ui_check():
    if not (UI / "package.json").exists():
        print("[update_state] no ui/package.json -- nothing to build")
        _clear(UI_REBUILD)
        return 0

    # No bundle at all -> must build.
    if not (UI / "dist").exists():
        print("[update_state] no ui/dist -- rebuild needed")
        _write(UI_REBUILD, "no dist\n")
        return 0

    head = _head()
    if not head:
        print("[update_state] could not read HEAD -- rebuilding to be safe")
        _write(UI_REBUILD, "unknown HEAD\n")
        return 0

    try:
        built = UI_BUILT.read_text(encoding="utf-8").strip()
    except OSError:
        print("[update_state] no build record -- rebuild needed")
        _write(UI_REBUILD, "no build record\n")
        return 0

    if built == head:
        print(f"[update_state] UI already built from {head[:12]} -- skipping rebuild")
        _clear(UI_REBUILD)
        return 0

    # Did anything under ui/ actually change between the two revisions?
    r = _git("diff", "--quiet", built, head, "--", "ui")
    if r.returncode == 0:
        print(f"[update_state] ui/ unchanged {built[:12]}..{head[:12]} -- skipping rebuild")
        # Advance the record so the next update compares against HEAD.
        _write(UI_BUILT, head + "\n")
        _clear(UI_REBUILD)
        return 0
    if r.returncode == 1:
        print(f"[update_state] ui/ changed {built[:12]}..{head[:12]} -- rebuild needed")
    else:
        # Unknown revision (history rewritten upstream), git error, etc.
        print("[update_state] could not diff ui/ -- rebuilding to be safe")
    _write(UI_REBUILD, "ui changed\n")
    return 0


def cmd_ui_done():
    head = _head()
    if head:
        _write(UI_BUILT, head + "\n")
        print(f"[update_state] UI build recorded at {head[:12]}")
    _clear(UI_REBUILD)
    return 0


COMMANDS = {
    "record": cmd_record,
    "ui-check": cmd_ui_check,
    "ui-done": cmd_ui_done,
}


def main(argv):
    if len(argv) != 2 or argv[1] not in COMMANDS:
        print(f"usage: {Path(argv[0]).name} {{{'|'.join(COMMANDS)}}}")
        return 2
    try:
        return COMMANDS[argv[1]]()
    except Exception as exc:
        # Never let bookkeeping break an update.
        print(f"[update_state] {argv[1]} failed, continuing: {exc}")
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
