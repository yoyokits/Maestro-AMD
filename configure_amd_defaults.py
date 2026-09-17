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

Setting `minimax_h3_text_encoder_default` also redirects the *download*:
wgp.py resolves the encoder's URLs from the selected variant
(`text_encoder_spec.get("URLs")`), so users fetch the ~14.6 GB GGUF
Q4_K_M file instead of the ~15.7 GB NVFP4 one.

Idempotent and conservative. Run it as often as you like.
"""
import json
import sys
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


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        _log(f"could not read {path.name}: {exc}")
        return None


def _write_json(path, data):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=4), encoding="utf-8")
        return True
    except OSError as exc:
        _log(f"could not write {path}: {exc}")
        return False


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


def apply_h3_default():
    """Override the H3 text-encoder default via finetunes/."""
    types = h3_model_types()
    if not types:
        _notes.append("no MiniMax H3 model definitions found -- nothing to do")
        return

    for model_type in types:
        target = FINETUNES_DIR / f"{model_type}.json"
        existing = _read_json(target)

        if target.exists() and not _is_managed(existing):
            # Someone else's finetune for this model type. Never clobber.
            _notes.append(f"{target.name}: user-authored, left alone")
            continue

        desired = _read_json(DEFAULTS_DIR / f"{model_type}.json")
        model = desired.get("model") if isinstance(desired, dict) else None
        if not isinstance(model, dict) or not model.get("architecture"):
            # Never write a copy that upstream's loader would crash on.
            _notes.append(
                f"defaults/{model_type}.json has no model.architecture -- skipped"
            )
            continue
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
        try:
            path.unlink()
            _changes.append(f"{path.name}: removed (no longer defined upstream)")
        except OSError as exc:
            _log(f"could not remove {path}: {exc}")


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

    remove_orphaned_h3_overrides()
    apply_h3_default()
    heal_h3_settings()
    heal_attention_mode()

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
