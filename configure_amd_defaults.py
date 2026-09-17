"""Seed AMD-appropriate defaults for settings whose stock values are
NVIDIA-specific.

Two settings ship with defaults that are wrong -- one of them dangerously
so -- on any AMD GPU:

1. **MiniMax H3 text encoder = NVFP4 AWQ ("Recommended").** NVFP4 is an
   NVIDIA-only quantization format. On AMD there is no kernel for it, so
   the 32B Qwen3-VL text encoder silently falls back to CPU while mmgp is
   already shuttling the ~20B transformer to and from disk. The combined
   RAM + I/O pressure pushes Windows into paging its own working set and
   the whole machine hangs -- not just Maestro, the OS. Recovery is a
   hard reset. See CLAUDE.md "Known runtime issues" #6.

   Upstream labels it "(Recommended)" on AMD because of a real logic gap
   in `_recommend_text_encoder()`: the first `nvfp4_awq` branch checks
   `hardware.get("supports_nvfp4")`, but the second one
   (`if "nvfp4_awq" in choices and ram_gb >= 24`) does not -- so any
   machine with >=24 GB RAM lands on NVFP4 regardless of whether the GPU
   can actually do it.

   **Scope of this override.** It replaces a default that can hang the
   machine. It is not a claim that H3 is fast on AMD -- H3 is a heavy
   model with a 32B encoder, and SLA sparse attention is unavailable
   without Triton, so it falls back to dense sdpa. But it does work:
   see the Q4_K_M note below.

2. **Attention mode.** Ships as `"auto"`, which resolves through
   `sage2 -> sage -> sdpa` against *installed* backends. On AMD none of
   the Sage variants are installed, so `"auto"` already lands on `sdpa`
   and is correct. It only needs fixing when a config has a stale
   explicit value (a user picked Sage in the UI, or carried a config over
   from an NVIDIA machine), which would otherwise fail at generation
   time.

**How the H3 fix is applied -- no upstream files are touched.** wgp.py
discovers model definitions from `defaults/*.json` **and**
`finetunes/*.json`, sorted, so a `finetunes/` file with the same basename
is processed after the `defaults/` one. `app/finetunes/*.json` is
git-ignored upstream, so the override survives `git reset --hard` in
update.js. That folder is the documented extension point -- upstream ships
a "put your finetunes here.txt" in it.

**The override must be a FULL copy of the defaults file, never a delta.**
Up to Maestro v2.1.6 a same-named finetune was merged over the default
(`existing_model_def.update(model_def)`), so a two-key override worked.
From v2.2.0 `load_model_definitions()` registers the raw finetune dict,
runs `init_model_def()` on it (which reads `model_def["architecture"]`),
then *replaces* the default (`existing_model_def.clear(); .update(...)`).
A partial file therefore crashes startup with `KeyError: 'architecture'`,
and a partial file that merely adds `architecture` would silently drop the
variant's name, URLs and default settings. See CLAUDE.md "Known runtime
issues" #12. The copy is rebuilt from the current defaults file on every
run, so it tracks upstream changes to those defaults.

The ownership marker lives inside `"model"`: under the v2.2.0 loader every
top-level key becomes a UI setting and would leak into
`settings/<model>_settings.json`.

**Never leave a file behind that stops Maestro from starting.** Upstream's
loader raises on *any* `finetunes/*.json` that is unparseable or lacks
`model.architecture`, whoever wrote it. So before anything else every such
file is removed (ours) or renamed to `<name>.json.disabled` (anyone else's,
never deleted); writes are atomic; and if a default ever stops looking
like one we understand, our copy is removed and upstream's stock default
wins. Each pass is isolated, and the script always exits 0.

Setting `minimax_h3_text_encoder_default` also redirects the *download*:
wgp.py resolves the encoder's URLs from the selected variant
(`text_encoder_spec.get("URLs")`), so users fetch the ~14.6 GB GGUF
Q4_K_M file instead of the ~15.7 GB NVFP4 one.

Idempotent and conservative. Run it as often as you like.
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
APP = REPO / "Maestro" / "app"
DEFAULTS_DIR = APP / "defaults"
FINETUNES_DIR = APP / "finetunes"
SETTINGS_DIR = APP / "settings"
WGP_CONFIG = APP / "wgp_config.json"

# Marks a finetunes file as written by this wrapper, so re-runs may update
# it but a file the *user* authored is never touched.
MARKER = "_maestro_amd_managed"

# Confirmed against app/models/minimax_h3/minimax_h3_handler.py
# (_text_encoder_variants): nvfp4_awq, gguf_q2_k, gguf_q4_k_m, int8, bf16.
#
# Q4_K_M is the balanced variant and is field-proven on AMD: a live
# RX 7900 XTX / 32 GB install ran nine H3 generations with it over two
# days at 6 steps, up to 52,781 packed rows, at acceptable speed.
#
# Q2_K (8.5 GB) is the lower-RAM option and stays a user choice, not the
# default -- it costs quality, and the evidence does not support paying
# that by default. Users switch in Advanced Settings -> "H3 Text
# Encoder"; the choice lands in settings/<model>_settings.json and wins
# over this default.
H3_SAFE_ENCODER = "gguf_q4_k_m"
H3_BAD_ENCODER = "nvfp4_awq"
H3_DEFAULT_KEY = "minimax_h3_text_encoder_default"
H3_SETTING_KEY = "minimax_h3_text_encoder"

# Attention modes that cannot work on AMD because the wrapper installs no
# Triton/Sage/FlashAttention/xformers. "auto" and "sdpa" are both fine.
ATTENTION_KEY = "attention_mode"
ATTENTION_SAFE = "auto"
ATTENTION_UNAVAILABLE = {"sage", "sage2", "sage3", "flash", "xformers", "radial", "sla"}

_changes = []
_notes = []


def _log(msg):
    print(f"[amd_defaults] {msg}")


MISSING, INVALID, OK = "missing", "invalid", "ok"


def _load_json(path):
    """(status, data) -- keeps "unreadable" distinct from "absent"."""
    try:
        return OK, json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return MISSING, None
    except (OSError, ValueError) as exc:
        return INVALID, exc


def _read_json(path):
    status, data = _load_json(path)
    return data if status == OK else None


def _write_json(path, data):
    # The tmp name must not end in .json, or upstream's loader would glob it.
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data, indent=4), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        _log(f"could not write {path}: {exc}")
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def _remove(path, why):
    try:
        path.unlink()
        _changes.append(f"{path.name}: removed ({why})")
    except FileNotFoundError:
        pass
    except OSError as exc:
        _log(f"could not remove {path}: {exc}")


def h3_model_types():
    """Every MiniMax H3 model type upstream currently defines.

    Discovered from defaults/ rather than hardcoded, so a new H3 variant
    upstream is covered automatically.
    """
    if not DEFAULTS_DIR.is_dir():
        return []
    return sorted(
        p.stem for p in DEFAULTS_DIR.glob("minimax_h3*.json") if p.is_file()
    )


def _is_managed(data):
    """True if this wrapper wrote the file.

    Current files carry the marker inside "model"; files written before
    Maestro v2.2.0 support carry it at top level and must still count, so
    they get rewritten instead of being mistaken for user files.
    """
    if not isinstance(data, dict):
        return False
    model = data.get("model")
    return bool(data.get(MARKER)) or (isinstance(model, dict) and bool(model.get(MARKER)))


def definition_problem(data):
    """Why upstream's loader would crash on this parsed definition, or None.

    Mirrors what `load_model_definitions()` dereferences unconditionally:
    `json_def["model"]` and then `model_def["architecture"]`.
    """
    if not isinstance(data, dict):
        return "not a JSON object"
    model = data.get("model")
    if not isinstance(model, dict):
        return "no \"model\" block"
    if not model.get("architecture"):
        return "no model.architecture"
    return None


def _fatal(status, data):
    if status == MISSING:
        return None
    if status == INVALID:
        return f"unreadable JSON ({data})"
    return definition_problem(data)


def fatal_reason(path):
    """Why Maestro would refuse to start on this finetunes file, or None."""
    return _fatal(*_load_json(path))


def loader_traits():
    """(strict, replaces) for the installed wgp.py's definition loader.

    strict: it indexes json_def["model"] / model_def["architecture"]
    directly, so their absence is a crash. replaces: a same-named finetune
    replaces its default (v2.2.0+) instead of merging over it (<= v2.1.x),
    so a partial override is a crash too. Read from the source so a user
    file is only ever renamed when the installed loader really rejects it.
    """
    try:
        source = (APP / "wgp.py").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, False
    strict = 'json_def["model"]' in source and 'model_def["architecture"]' in source
    return strict, "existing_model_def.clear()" in source


def _loader_accepts(path, data, traits):
    """Whether a parseable file flagged by definition_problem still loads."""
    strict, replaces = traits
    if not strict:
        return True
    partial_override = (
        isinstance(data, dict)
        and isinstance(data.get("model"), dict)
        and (DEFAULTS_DIR / path.name).is_file()
    )
    return partial_override and not replaces


def _quarantine_name(path):
    target = path.with_name(path.name + ".disabled")
    if target.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = path.with_name(f"{path.name}.disabled-{stamp}")
    return target


def quarantine_fatal_finetunes():
    """Move every finetune that would crash Maestro at startup out of the way.

    Ours are deleted (apply_h3_default rebuilds them). Anything we can't
    prove we wrote -- including unparseable files -- is renamed, never
    deleted, so a user's work survives.
    """
    if not FINETUNES_DIR.is_dir():
        return
    traits = None
    for path in sorted(FINETUNES_DIR.glob("*.json")):
        status, data = _load_json(path)
        reason = _fatal(status, data)
        if reason is None:
            continue
        if status == OK and _is_managed(data):
            _remove(path, f"outdated wrapper override, {reason}")
            continue
        if status == OK:
            traits = traits or loader_traits()
            if _loader_accepts(path, data, traits):
                _notes.append(f"{path.name}: {reason}, but this Maestro's loader "
                              "accepts it -- left alone")
                continue
        target = _quarantine_name(path)
        try:
            os.replace(path, target)
        except OSError as exc:
            _log(f"could not move aside {path}: {exc}")
            continue
        _changes.append(
            f"{path.name}: {reason} -- Maestro would crash at startup on it. "
            f"Renamed to {target.name}; fix it and rename it back to "
            f"{path.name}."
        )


def apply_h3_default():
    """Override the H3 text-encoder default via finetunes/."""
    types = h3_model_types()
    if not types:
        _notes.append("no MiniMax H3 model definitions found -- nothing to do")
        return

    for model_type in types:
        target = FINETUNES_DIR / f"{model_type}.json"
        status, existing = _load_json(target)

        if status != MISSING and not (status == OK and _is_managed(existing)):
            # Someone else's valid finetune (fatal ones were already moved).
            _notes.append(f"{target.name}: user-authored, left alone")
            continue

        desired = _read_json(DEFAULTS_DIR / f"{model_type}.json")
        problem = definition_problem(desired)
        if problem:
            # A default we don't understand: drop our copy and let upstream's
            # own default load, rather than risk a stale or broken override.
            _notes.append(f"defaults/{model_type}.json: {problem} -- not overridden")
            if status == OK:
                _remove(target, "its upstream default changed shape")
            continue
        model = desired["model"]
        model[H3_DEFAULT_KEY] = H3_SAFE_ENCODER
        model[MARKER] = True

        if existing == desired:
            continue
        if _write_json(target, desired):
            _changes.append(
                f"{model_type}: H3 text-encoder default -> {H3_SAFE_ENCODER}"
            )


def remove_orphaned_h3_overrides():
    """Delete managed H3 overrides whose upstream default no longer exists.

    A full copy outlives its source, so without this a variant removed
    upstream would keep appearing in the UI with stale URLs.
    """
    if not FINETUNES_DIR.is_dir():
        return
    for path in sorted(FINETUNES_DIR.glob("minimax_h3*.json")):
        if (DEFAULTS_DIR / path.name).is_file():
            continue
        if not _is_managed(_read_json(path)):
            continue
        _remove(path, "no longer defined upstream")


def heal_h3_settings():
    """Fix saved per-model settings that still select NVFP4.

    Only the known-hang value is touched. A user who deliberately picked
    GGUF Q2_K (less RAM) keeps it.
    """
    if not SETTINGS_DIR.is_dir():
        return
    for path in sorted(SETTINGS_DIR.glob("minimax_h3*_settings.json")):
        data = _read_json(path)
        if not isinstance(data, dict):
            continue
        if data.get(H3_SETTING_KEY) != H3_BAD_ENCODER:
            continue
        data[H3_SETTING_KEY] = H3_SAFE_ENCODER
        if _write_json(path, data):
            _changes.append(
                f"{path.name}: {H3_BAD_ENCODER} -> {H3_SAFE_ENCODER} "
                "(this setting hangs the OS on AMD)"
            )


def heal_attention_mode():
    """Reset an attention mode that cannot run on this install.

    Healed to "auto", not "sdpa": auto already resolves to sdpa when no
    Sage/Flash backend is installed, and it will pick up a faster backend
    automatically if one is ever added (see PERFORMANCE.md section 3.5).
    """
    data = _read_json(WGP_CONFIG)
    if not isinstance(data, dict):
        return
    current = data.get(ATTENTION_KEY)
    if current is None or current not in ATTENTION_UNAVAILABLE:
        return
    data[ATTENTION_KEY] = ATTENTION_SAFE
    if _write_json(WGP_CONFIG, data):
        _changes.append(
            f"wgp_config.json: attention_mode {current!r} is not installed "
            f"on AMD -> {ATTENTION_SAFE!r} (resolves to sdpa)"
        )


def main():
    if not APP.is_dir():
        _log(f"no Maestro install at {APP} -- skipping")
        return 0

    # Isolated so one unexpected failure can't skip the passes after it --
    # and this helper must never be the thing that blocks Start.
    for step in (
        quarantine_fatal_finetunes,
        remove_orphaned_h3_overrides,
        apply_h3_default,
        heal_h3_settings,
        heal_attention_mode,
    ):
        try:
            step()
        except Exception as exc:
            _log(f"{step.__name__} failed: {exc!r}")

    for note in _notes:
        _log(note)
    if _changes:
        for change in _changes:
            _log(change)
    else:
        _log("AMD defaults already in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
