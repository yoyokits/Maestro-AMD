# Maestro AMD — repo guide for Claude Code

## What this is

A **minimal Pinokio wrapper** around upstream
[Blizaine/Maestro](https://github.com/Blizaine/Maestro), specialized for
AMD GPUs. Upstream Maestro is an all-in-one local AI video/image/music
studio built on the WanGP pipeline. The stock upstream is NVIDIA-oriented;
this wrapper adds AMD ROCm support without vendoring or forking upstream
code — install/update just `git clone` / `git pull` upstream fresh.

Do **not** copy upstream Python source into this repo. Do **not** patch
upstream from here. If upstream needs a change, send a PR upstream. The
whole point of the wrapper shape is that upstream stays authoritative.

A **sibling project** (repo TBD) covers NVIDIA. Do not add NVIDIA branches
to this codebase; if a genuinely shared abstraction emerges, propose a
third repo both wrappers depend on.

## Layout

Repo root (after `Install` has been clicked in Pinokio):

```
Maestro-AMD/
├── pinokio.js              ← app metadata + dynamic menu
├── install.js              ← clone + build (AMD-guarded, fails fast)
├── torch.js                ← ROCm wheel install, per gfx target
├── start.js                ← daemon: python launch.py (+ user_env.json)
├── update.js               ← git pull + refresh deps
├── start_latest.js         ← "Update & Start" one-click
├── repair.js               ← rebuild the env, KEEP models
├── rollback.js             ← undo the last update
├── reset.js                ← rm -rf Maestro/ (destroys models)
├── diagnose.js/.py         ← health check; run this first on any report
├── sam_install.js          ← optional Inpaint / SAM 3.1 (isolated env)
├── launcher_profile.js     ← GPU detection + runtime profile
├── maestro_amd_preamble.py ← FSDP shadow, see "Known runtime issues" #1
├── install_preamble.py     ← installs the above + its .pth into the venv
├── verify_rocm_torch.py    ← guards against a CUDA/CPU torch, see #7
├── configure_amd_defaults.py ← AMD-correct H3/attention defaults, see #6
├── update_state.py         ← rollback point + UI-rebuild bookkeeping
├── ensure_ffmpeg.py        ← provisions ffmpeg/ffprobe, see #3/#4
├── test_gpu_detection.js   ← unit tests for launcher_profile/torch.js detection logic
├── test_amd_defaults.py    ← finetune-override tests + upstream loader contract, see #12
├── .maestro_state/         ← wrapper state (git-ignored, OUTSIDE Maestro/)
├── user_env.json           ← optional env overrides (git-ignored)
├── CLAUDE.md               ← this file
├── PERFORMANCE.md          ← CUDA-vs-ROCm gap analysis, what's closable
├── README.md               ← user-facing (short)
└── Maestro/                ← upstream clone (created by install.js)
    ├── app/                ← Python backend (launch.py, wgp.py, ...)
    │   ├── env-amd/        ← Python 3.11 venv with ROCm torch
    │   ├── models/         ← downloaded checkpoints (git-ignored)
    │   ├── loras/          ← user LoRAs (git-ignored)
    │   ├── outputs/        ← generated video/image (git-ignored)
    │   └── wgp_config.json ← user settings (git-ignored)
    └── ui/                 ← React frontend (npm install + build)
```

Everything under `Maestro/app/` that is user data is `.gitignore`'d
upstream, which is why `git pull` on update naturally preserves it.

## AMD GPU support

`launcher_profile.js` is the **single source of truth** for what's
supported. Adding a GPU family means editing three places together:

1. Add a helper (`isAmdFoo`) in `launcher_profile.js`.
2. Add the family's gfx targets to `WIN_WHEEL_INDEXES` in `torch.js`
   (and the fallback table `GPU_NAME_TARGETS` in `launcher_profile.js`
   if Pinokio's own detection can't name them — see "GPU detection").
3. Add matching assertions in `test_gpu_detection.js`.

Currently supported (confirmed August 2026):

| Family              | GFX targets                     | Products                          |
|---------------------|---------------------------------|-----------------------------------|
| RDNA 2 dGPU         | gfx1030, gfx1031, gfx1032, gfx1034 | RX 6000 series                 |
| RDNA 3 dGPU         | gfx1100, gfx1101, gfx1102       | RX 7000/8000 (incl. 7900 XTX)     |
| RDNA 4 dGPU         | gfx1200, gfx1201                | RX 9000 series                    |
| Strix Point APU     | gfx1150                         | Ryzen AI 300 (Radeon 880M/890M)   |
| Strix Halo APU      | gfx1151                         | Ryzen AI Max (8060S)              |
| Krackan Point APU   | gfx1152                         | Ryzen AI Krackan Point            |
| Krackan Halo APU    | gfx1153                         | Ryzen AI Krackan Halo             |

## Library versions

Pinned in `torch.js`. Bump procedure:

- **Linux (stable)** — currently `torch==2.11.0` / `torchvision==0.26.0` /
  `torchaudio==2.11.0` via `https://download.pytorch.org/whl/rocm7.2`. To
  bump: check <https://pytorch.org/get-started/previous-versions/> and
  verify the exact triple exists at
  `https://download.pytorch.org/whl/rocm<new_ver>/`. All three packages
  must match ABI (they release together — never mix versions).
- **Windows (nightly)** — unpinned by design. Pulled from
  `https://rocm.nightlies.amd.com/v2-staging/<target>` at install time.
  If AMD reorganizes their v2-staging tree (new/removed target
  directories), update `WIN_WHEEL_INDEXES` in `torch.js`, the helper
  regexes in `launcher_profile.js`, and `test_gpu_detection.js`
  together. Cross-check against
  <https://rocm.nightlies.amd.com/v2-staging/>.
- **Python** — 3.11. ROCm wheels do not ship cp310 builds. Do not
  downgrade.

Skipped intentionally (all CUDA-only, no ROCm equivalent): SageAttention,
FlashAttention, triton-windows, nunchaku, lightx2v, xformers. WanGP's
built-in PyTorch SDPA path handles attention on AMD — but see "Known
runtime issues" #5: SDPA needs an env var set at launch or it silently
picks the OOM-prone fallback kernel.

## GPU detection

`torch.js` needs a GFX target to pick the Windows wheel index. Pinokio
exposes one (`gpu_target` in the template environment, `kernel.gpu_target`
in JS), resolved by pinokiod's `kernel/gpu/amd_gfx_targets.json` name
table. **That table is incomplete for real device names** — verified
August 2026, pinokiod 8.0.40 and 8.0.118 alike:

- Strix Halo APUs on Windows report their iGPU as
  `AMD Radeon(TM) 8060S Graphics`. The trailing "Graphics" (Windows'
  convention for APU iGPU display names) defeats pinokiod's exact-match
  lookup: the table has `radeon 8060s` but not `radeon 8060s graphics`,
  and pinokiod's CPU-brand fallback only fires for the literal string
  "radeon graphics". Result: `gpu_target = null`.
- pinokiod 8.0.118's new PCI-ID fallback (`amd_pci_targets.json`) also
  misses Strix Halo: device `1002:1586` is not in its table.

With `gpu_target` null, the original template-`when`-gated `torch.js`
skipped **every** wheel-install step and exited at the "unsupported"
notify, never writing the runtime marker — which surfaces to the user as
"AMD ROCm runtime not installed" at Start. The author's reference
machine was an RX 7900 XTX dGPU, whose WMI name carries no trailing
"Graphics", which is why this shipped undetected.

**Fix — `resolveGpuTarget()` in `launcher_profile.js`.** Trusts Pinokio's
`gpu_target` when it's a valid gfx id; otherwise resolves from the raw
`kernel.gpu_model` / `kernel.gpus[*].model` strings via the
`GPU_NAME_TARGETS` fallback table (APU iGPU names first, then dGPU
family patterns). `torch.js` calls it in JS and builds its run steps
from the resolved target — the template `gpu_target` variable is no
longer load-bearing. `isAmdRdna2/3/4` and `isAmdApu` route through the
same resolver, so they work on affected machines too.

If pinokiod ever fixes its tables, `resolveGpuTarget` keeps working
unchanged (it prefers Pinokio's answer). Keep `GPU_NAME_TARGETS` in sync
with `WIN_WHEEL_INDEXES` in `torch.js` and with the supported-family
table above. Unit tests in `test_gpu_detection.js`.

## Venv layout pitfall (`path` resolves `venv`)

Pinokio resolves the `venv` param **relative to a step's `path`**
(pinokiod `kernel/shell.js`: `env_path = path.resolve(params.path,
params.venv)`). This has one load-bearing consequence: **every step that
touches the venv must use `path: runtime.path` (`Maestro/app`)**, or
Pinokio activates/creates a different venv at whatever directory you
named instead.

The pre-fix wrapper ran the two root-level helper scripts
(`install_preamble.py`, `ensure_ffmpeg.py`) with `path: "."`. That
created a stray `env-amd` at the **app root**, and the preamble +
`ffmpeg`/`ffprobe` landed there — while the real runtime venv at
`Maestro/app/env-amd` (used by `torch.js` and `start.js`, which correctly
use `path: "Maestro/app"`) never got the FSDP shadow. Result: `launch.py`
crashed on the `torch.distributed.fsdp` import — the exact symptom of
Known runtime issue #1, despite the fix being "installed".

**Fix.** Helper steps now use `path: runtime.path` and invoke the helper
by absolute path (`python "${__dirname}/install_preamble.py"`), so
the venv stays the runtime one while the script still resolves from the
wrapper root. `reset.js` also removes the stray root `env-amd` so a Reset
returns to a truly clean state. `runtime.path` is baked into the
`torch.js` wheel-step `path` templates as a literal (`'${runtime.path}'`)
because Pinokio template memory has no `runtime` binding — an unresolved
`runtime.path` in a template is a bug, and the unit test asserts it never
appears.

If you add a new venv-touching step, `path: runtime.path` is the rule. A
step with any other `path` gets its own private venv.

## How updates work

`update.js` runs, in order:

1. Records the current upstream SHA to `.maestro_state/prev_head` so
   `rollback.js` has somewhere to go.
2. `git -C Maestro fetch origin && git -C Maestro reset --hard origin/HEAD`
   — matches upstream tracked files exactly. **Wipes any manual edits**
   inside `Maestro/`. Untracked user data (models, outputs, config) is
   preserved because `git reset` only touches tracked files.
3. Re-installs the startup preamble (also migrates pre-`.pth` installs).
4. Re-runs `uv pip install -r requirements.txt` in the venv.
5. Runs `verify_rocm_torch.py`, which flags a reinstall if the resolver
   replaced the ROCm torch with a CUDA or CPU build (issue #7).
6. Runs `torch.js` **only if** the runtime marker
   (`Maestro/app/env-amd/.maestro_amd_v1.installed`) is missing **or**
   that flag is set. Delete the marker to force a full ROCm wheel
   reinstall on the next update.
7. Provisions ffmpeg/ffprobe, then re-asserts the AMD-correct defaults
   (issue #6) — upstream may have added new model variants needing them.
8. Self-heals the `seedvc` component if the user deleted it.
9. Rebuilds the React UI **only when `ui/` actually changed** between the
   old and new revision (`update_state.py ui-check`). Fails open.

For **easy/automatic updating** the wrapper exposes two shapes in the
Pinokio menu:

- **Update** — just runs `update.js`.
- **Update & Start** — runs `start_latest.js`, which chains `update.js`
  → `start.js`. One click, always latest. This is the recommended
  everyday launcher.

Recovery ladder, cheapest first — the menu presents them in this order
and only offers Reset last:

- **Diagnose** (`diagnose.js`) — read-only health report. Always the
  first thing to ask a user for.
- **Roll back last update** (`rollback.js`) — only shown when
  `.maestro_state/prev_head` exists. Returns to the pre-update revision.
- **Repair** (`repair.js`) — `git clean -fdx` inside `Maestro/` with the
  user-data paths excluded, then a full reinstall. Keeps models.
- **Reset** (`reset.js`) — deletes everything, models included.

`.maestro_state/` deliberately lives *outside* `Maestro/` so Repair's own
`git clean` cannot destroy the rollback point.

## Development / testing changes

Pinokio scripts cannot be exercised from a shell — you have to drive
Pinokio itself. To try a change:

1. Save the file (Pinokio hot-reloads scripts).
2. Point Pinokio at this folder if it isn't already.
3. Click **Install** (fresh) or the specific action you changed.
4. Watch the terminal panel for command output; watch the browser
   devtools console for menu/UI issues.

Syntax-check locally before iterating in Pinokio:

```
node --check pinokio.js
node --check install.js
node --check torch.js
node --check start.js
node --check update.js
node --check start_latest.js
node --check reset.js
node --check launcher_profile.js
node --check test_gpu_detection.js
```

`test_gpu_detection.js` exercises `resolveGpuTarget` (including the
Strix Halo failing-machine profile) and the `torch.js` step shapes for
supported/unsupported/unknown targets on both platforms. Run it after
touching either file: `node test_gpu_detection.js`.

`test_amd_defaults.py` covers `configure_amd_defaults.py` (issue #12) and
runs upstream's real finetune loader over its output. Stdlib only, no
torch — any Python 3 works:

```
python test_amd_defaults.py --wgp Maestro/app/wgp.py [--app Maestro/app]
```

`--wgp` is repeatable (point it at older/newer upstream `wgp.py` files to
check several loaders); `--app` seeds the contract with a real install's
defaults and finetunes (read-only).

For the async-export files (install/torch/start/update),
`node --check` catches parse errors but not runtime issues in the
returned config object. Test the runtime path via Pinokio.

The Python helpers (`maestro_amd_preamble.py`, `install_preamble.py`,
`ensure_ffmpeg.py`, `diagnose.py`, `verify_rocm_torch.py`,
`update_state.py`, `configure_amd_defaults.py`)
are plain Python — check with
`python -m py_compile <them>`
using the same `env-amd` venv install.js/update.js invoke them with, then
actually run them against a real `Maestro/` clone (all are idempotent and
safe to re-run) before trusting a change to any. For the preamble
in particular, verify by starting a fresh Python interpreter in the venv
and checking `'torch.distributed.fsdp' in sys.modules` — it must be
`True` before any user code runs.

## Known runtime issues & fixes (first bring-up, RX 7900 XTX / Windows, Aug 2026)

Real problems hit getting Maestro AMD running end-to-end on Windows, and
what fixed each one. Kept here so nobody re-derives this from scratch.

### 1. Crash on every Start: `ModuleNotFoundError: torch._C._distributed_c10d`

**Symptom:** `launch.py` crashes at import time, before any UI loads:
```
File "...\models\wan\distributed\fsdp.py", line 5, in <module>
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
...
ModuleNotFoundError: No module named 'torch._C._distributed_c10d'
```

**Root cause:** upstream Maestro's `models/wan/distributed/fsdp.py` does an
unconditional top-level `from torch.distributed.fsdp import
FullyShardedDataParallel as FSDP`. `any2video.py` imports `shard_model`
from that file but **never calls it** anywhere on the single-GPU inference
path — dead code. The AMD ROCm-for-Windows nightly wheel ships without
RCCL/GLOO, so `torch.distributed.is_available()` is `False` and that import
chain crashes, taking the whole app down for a code path nothing uses.
Confirmed this isn't ROCm-specific in principle — any PyTorch build without
a working distributed backend would hit it.

**Fix — automated, `maestro_amd_preamble.py` + `install_preamble.py`.**
Zero modification to Maestro's tree. `install_preamble.py` (invoked by
`install.js`/`update.js` after every clone/reset) drops
`sitecustomize.py` into the venv's `site-packages` — a filename CPython
auto-loads at every interpreter startup, before any user code runs. It
preemptively writes fake `torch.distributed.fsdp` and
`torch.distributed.fsdp.wrap` modules to `sys.modules`. When Maestro's
`from torch.distributed.fsdp import ...` runs later, Python finds our
fakes in the module cache and skips loading the real crashing module
entirely.

The fake exposes exactly what Maestro reads at def time
(`ShardingStrategy.FULL_SHARD`, plus callable placeholders for `FSDP`,
`MixedPrecision`, `lambda_auto_wrap_policy`). Anything actually invoked
raises a clear `RuntimeError` — never happens on the single-GPU path,
because `shard_model()` is never called.

Lives in the venv, so **Update preserves it for free** — no source patch
to reapply, no `git reset --hard` interaction. Only a `Reset` (which
wipes the whole `Maestro/` folder, venv included) would require redoing
it; `install.js` handles that on the next Install.

**Earlier approaches, why abandoned:**

- A **source patch** to `Maestro/app/models/wan/distributed/fsdp.py`
  (was the first version here — see git history for `patch_fsdp.py`).
  Works, but has to re-apply after every `git reset --hard` in
  `update.js` (since that wipes any local edit to a tracked file), and
  it modifies Maestro's tree which is against the wrapper's design
  principle ("Do not vendor upstream" above). `sitecustomize.py`
  achieves the same result without touching Maestro at all.
- A **runtime import probe** (e.g. `sitecustomize.py` that did `try:
  import torch.distributed.fsdp` to decide whether to install the stub).
  That was tried first here and can trigger a runaway subprocess storm —
  see #2 below. **`sitecustomize.py` must NEVER import
  `torch.distributed.fsdp` or anything that transitively runs it** —
  write to `sys.modules` unconditionally instead.
- `sys.modules['torch.distributed.fsdp'] = None`. That signals to
  Python's import machinery that the module doesn't exist and turns
  future `from torch.distributed.fsdp import X` into `ImportError`,
  which Maestro doesn't handle. A fully-populated fake module is what
  actually works.

Filed upstream too: issue drafted against Blizaine/Maestro with the root
cause + fix (not auto-submitted — GitHub issues need the reporter's own
account). **If it's merged when you read this, delete
`maestro_amd_preamble.py`, `install_preamble.py`, and their call sites in
`install.js`/`update.js`** — the shadow shadows the *fixed* module too,
which is a real (if minor) footgun for anyone who ever wants working
multi-GPU FSDP downstream.

### 2. Do NOT let `torch.distributed.fsdp/__init__.py` run on this build

Any code that causes `torch.distributed.fsdp/__init__.py` to execute on
this ROCm-for-Windows nightly wheel can trigger a runaway subprocess
storm — hundreds of `offload-arch.exe`/`python.exe` processes spawned in
a loop, confirmed twice with wildly different severity from identical
code, including one case where a watchdog killed the first wave but a
second, larger wave followed from the same still-running root process.
This is a race/bug in AMD's ROCm/TheRock Windows toolchain
(`offload-arch` is a Python console-script invoked via subprocess for HIP
arch detection; see
[ROCm/TheRock#3262](https://github.com/ROCm/TheRock/issues/3262) and
[#5003](https://github.com/ROCm/TheRock/issues/5003)), not anything in
this repo.

The fix in issue #1 preempts this by shadowing `torch.distributed.fsdp`
in `sys.modules` *before* the real `__init__.py` can run. That's the safe
shape. Concretely, do not:

- `import torch.distributed.fsdp` (obvious).
- Import anything under `torch.distributed.fsdp.*` (submodule access
  triggers parent import).
- Call `torch.distributed.is_available()` — probably safe (the module
  import itself is), but if you must, guard with a process-count
  watchdog primed first (a continuous kill-on-threshold loop against
  `offload-arch.exe`, not a one-shot).

If you're debugging in this area, prime a continuous watchdog first —
one-shot watchdogs have been observed to miss follow-on waves. See git
history for `scratchpad/watchdog2.ps1` if a template is useful.

### 3. UI warning: "ffmpeg not found in path"

**Root cause:** `imageio-ffmpeg` (a real `requirements.txt` dependency)
bundles its own `ffmpeg.exe` inside its site-packages install, but that's
not on `PATH` under the name `ffmpeg`. Separately, `ffmpeg-python`/`ffmpy`
and Gradio's video-preview code shell out to a literal `ffmpeg`/`ffprobe`
on `PATH` — they don't know about imageio-ffmpeg's private copy.

### 4. Follow-on crash: `ffprobe` not found

**Root cause:** `imageio-ffmpeg` bundles `ffmpeg` only, not `ffprobe`.
Gradio's `video_is_playable()` check needs `ffprobe` specifically (one
plugin's tutorial-video preview failed on this — non-fatal, caught and
logged per-plugin, did not crash the app, but should still be fixed).

**Fix for both — now automated, `ensure_ffmpeg.py`.** Downloads a matched
ffmpeg+ffprobe pair (same binary source the `static-ffmpeg` PyPI package
uses: `github.com/zackees/ffmpeg_bins`, predictable per-platform zip
layout, no versioned-folder-name guessing) and copies both next to the
active venv's own Python executable — which is already on `PATH` whenever
Pinokio's `venv` param is used, i.e. every real Start/Update/Install. Skips
entirely once both binaries already exist (cheap, idempotent, self-healing
if they ever go missing). `install.js` runs it right after the pip-install
step; `update.js` runs it every time too (still a no-op on a cache hit —
the binaries live in the venv, which `Update` never touches, so in
practice this only does real work once per install).

**Important implementation detail — do not use `static-ffmpeg`'s own
Python downloader, and do not use Python's `requests` for fetches in this
repo's helper scripts at all.** `static-ffmpeg` itself downloads via
`requests`, which failed here with
`SSLCertVerificationError: unable to get local issuer certificate` against
a plain `github.com` URL — even though `curl` and `git` on the exact same
machine, same network, succeeded immediately. This is the classic
signature of a Python venv whose `certifi` bundled root-CA list doesn't
include something the OS's own trust store does (e.g. behind a
corporate/AV TLS-inspection proxy) — a real, observed failure mode for
uv-managed Windows Python interpreters, not a one-off. `ensure_ffmpeg.py`
shells out to `curl` for the download and only uses Python's stdlib
`zipfile` (pure local file I/O, no TLS involved) to extract — no
`requests`, no extra pip dependency. If you add another script that needs
to fetch something, follow the same pattern.

### 5. OOM crash on generation: `HIP out of memory. Tried to allocate 92.55 GiB`

**Symptom:** generation starts, runs the progress bar for a while, then
crashes mid-denoising inside `F.scaled_dot_product_attention` with a HIP
OOM error requesting far more memory than the GPU has (tens of GB on a
24GB card), on models/settings with a large packed sequence length (long
videos, high frame counts, large resolutions).

**Root cause:** PyTorch's SDPA has three backends — Flash Attention,
Memory-Efficient Attention, and Math (naive fallback). On this ROCm
nightly, Flash and Memory-Efficient are implemented via AOTriton but
gated behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (unset by
default → both report "runtime disabled" and PyTorch silently falls back
to Math). Math materializes the **full O(n²) attention matrix** — at
Maestro's typical packed sequence lengths (tens of thousands of tokens
for a multi-second video) that's tens to hundreds of GB, far past any
consumer card. Confirmed live: the exact sequence length from a real
crash (21,063 rows) needed 92.55 GiB under Math; under Flash or
Memory-Efficient (same GPU, same shapes, env var set) peak usage was
under 150MB.

**Fix — `start.js` sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1"`**
in the env passed to `python launch.py`. Zero Maestro modification — it's
a PyTorch/ROCm runtime flag read at kernel-selection time, nothing
Maestro's code touches. "Experimental" per AMD's own warning, but the
alternative today is a guaranteed OOM on realistic generation lengths, so
enabling it is the correct default for a video-gen app. If AMD
stabilizes/defaults this in a future ROCm build, this becomes a no-op and
can be left in place or removed.

### 6. System hang (not just a crash) when generating with MiniMax H3

**Symptom:** loading a MiniMax H3 model (Omni, First/Last, Full, etc.)
shows heavy disk activity and near-zero GPU usage during text-encoder /
token-decoding, then generation freezes at step 0 and the entire OS
becomes unresponsive — no clean recovery short of a hard reset.

**Root cause:** every MiniMax H3 variant's text encoder defaults to
**NVFP4 AWQ** — an NVIDIA-only quantization format — regardless of GPU
vendor, and the UI labels it "(Recommended)". On AMD there's no kernel
for it, so PyTorch silently falls back to running the 32B-parameter
Qwen3-VL text encoder on **CPU**. Combined with mmgp's partial-pinning /
async-disk-shuttle offload path for the ~20B transformer (visible in the
console as `Switching to partial pinning...` and `Async loading plan...`
messages), this creates simultaneous heavy disk I/O + heavy RAM pressure
right as generation starts — enough to push Windows itself into paging
its own working set to disk, which is what actually causes the hang
(not just Maestro, the OS scheduler starves too).

Traced the "(Recommended)" mislabel to a real upstream logic gap in
`models/minimax_h3/minimax_h3_handler.py`'s `_recommend_text_encoder()`:
the first `nvfp4_awq` branch correctly checks
`hardware.get("supports_nvfp4")`, but a second `nvfp4_awq` fallback
branch (`if "nvfp4_awq" in choices and ram_gb >= 24`) does not — so any
machine with ≥24GB RAM (AMD included) still lands on NVFP4 regardless of
actual hardware support.

**Fix — user-facing, not wrapper-automatable.** This is a per-generation
UI setting inside Maestro itself (`minimax_h3_text_encoder`, exposed in
the React UI's Advanced Settings as **"H3 Text Encoder"**), not something
`install.js`/`torch.js` touches — there's nothing for the AMD wrapper to
patch here. Documented in README.md's "Before using MiniMax H3" section
for users: switch the dropdown from NVFP4 AWQ to **GGUF Q4_K_M** (or
**GGUF Q2_K** for less RAM). Both run correctly on CPU without the
NVFP4-specific fallback path's memory profile.

Separately, **Settings → Services → LLM Device: CPU** (the Director
planning LLM's device) is correct for every user regardless of GPU
vendor — it's not an AMD-specific fix, just worth confirming it's set
since GPU contention compounds the H3 problem above.

Worth filing upstream (the missing `supports_nvfp4` check in the second
branch): would fix the mislabeled "(Recommended)" on any non-NVFP4-capable
hardware, not just AMD.

### 6b. Two corrections to #6, from field evidence

**(a) NVFP4's fallback runs on the GPU, not the CPU.** #6 above says the
encoder "silently falls back to running on CPU". That is not what the
code does. `shared/qtypes/nvfp4.py`, in the branch that prints
`NVFP4: linear fallback`:

```python
qweight = weight.dequantize(dtype=input.dtype, device=input.device)
return torch.nn.functional.linear(input, qweight, bias=bias_arg)
```

It dequantizes **to `input.device`** — the GPU — and runs `F.linear`
there. It is slower than the fused NVFP4 kernel (a dequant per call) and
it materializes full-precision weights, which raises **memory** pressure.
The observed hang is best explained by that memory pressure on a heavy
model, not by CPU execution. Keep the override — the hang was real — but
do not repeat the "runs on CPU" explanation.

**(b) GGUF Q4_K_M is field-proven on AMD; do not claim otherwise.** Logs
from a live RX 7900 XTX / 32 GB install show **nine** H3 generations with
GGUF Q4_K_M across two days, at 6 steps, up to 52,781 packed rows, at
acceptable speed. An earlier revision of this file claimed H3 was
"impractical on AMD" and switched the default to Q2_K. That was wrong,
built on a single stalled run, and has been reverted.

**What that stalled run actually implicates.** Every slow/stalled
observation was on `minimax_h3_*_fused_turbo` at **4 steps** — upstream
labels it *"H3 Fused 4-Step — References (Experimental)"*, and the
checkpoint name (`...turbo8...`) suggests it is distilled for 8. The
successful runs were all at 6 steps on the non-fused H3. Before blaming
AMD for an H3 problem, **check the step count and whether the fused
turbo variant is selected** — that is the variable that actually
correlates.

**Method note.** The mistake was generalizing from one run without
checking `logs/api/start.js/*` for prior successes. Those logs record
the encoder loaded, step count and packed rows per generation:

```
grep -oE "Loading Text Encoder '[^']*'" logs/api/start.js/<id>
grep -oE "[0-9]+ steps; attention" logs/api/start.js/<id>
```

Read them before concluding anything about H3 performance.

### 7. `AssertionError: Torch not compiled with CUDA enabled`

**Symptom:** Maestro imports fine, logs some harmless-looking warnings,
then dies ~40 frames deep with a message unrelated to the real cause:

```
[Runtime] Python 3.11.15 | PyTorch 2.14.0+cpu | CUDA none | CUDA unavailable (unknown)
...
  File "...\models\wan\modules\t5.py", line 630, in T5EncoderModel
    device=torch.cuda.current_device(),
AssertionError: Torch not compiled with CUDA enabled
```

**The tell is `PyTorch 2.14.0+cpu`** — a stock PyPI wheel, not the ROCm
build. On ROCm, `torch.cuda.*` is the HIP API and works; on a CPU-only
wheel there is no `cuda` module, so the first `torch.cuda.current_device()`
in upstream's code asserts.

Note what is *not* the problem, because these lines mislead:
`Triton=missing`, `SageAttention=missing`, `FlashAttention is
unavailable` and `[GGUF][llama.cpp CUDA] kernels unavailable` are all
**normal on AMD** (PERFORMANCE.md §2) and appear on healthy installs too.

**Root cause:** nothing in `requirements.txt` pins torch, but
`torchcodec` / `accelerate` / `peft` / `timm` / `open_clip_torch` all
depend on it. If the resolver decides the installed ROCm wheel doesn't
satisfy a constraint it installs a stock PyPI torch over the top — CPU-only
on Windows. Especially easy to reach via **Update**, which used to skip
`torch.js` whenever the runtime marker existed, so nothing put the ROCm
wheel back.

**Fix — three layers, all automated:**

1. `install.js` installs the ROCm wheels **before** `requirements.txt`, so
   torch is already satisfied when the resolver runs.
2. `verify_rocm_torch.py` runs after every requirements pass and writes
   `.torch_needs_reinstall`; `update.js` re-runs `torch.js` on a missing
   marker **or** that flag. This is the self-heal — an affected user just
   clicks Update.
3. `verify_rocm_torch.py --preflight` in `start.js` catches it before
   launch and shows a readable dialog naming the cause and the fix.

The preflight is deliberately **metadata-only and conservative**: it never
imports torch, and only flags version strings that *prove* the build is
wrong (`+cpu`, `+cu`, `cuda`). An unrecognised string — AMD's Windows
nightlies do not always carry a `+rocm` tag — is treated as fine. A false
positive would block a working install, which is worse than a late
failure. Both flag files sit next to the venv and clear themselves once a
good build is seen, so they cannot go stale.

### 8. Update fails: `invalid peer certificate: UnknownIssuer` (uv)

**Symptom:** `uv pip install -r requirements.txt` dies partway through on
a wheel fetched by direct URL:

```
  x Failed to download `smplfitter @ https://github.com/.../smplfitter-0.2.10-py3-none-any.whl`
  |-> client error (Connect)
  `-> invalid peer certificate: UnknownIssuer
```

...while `git fetch` against the same host, in the same shell, succeeds.

**Root cause:** uv validates TLS against its own bundled webpki roots
rather than the OS trust store. Behind a corporate/AV TLS-inspection proxy
the intercepting root is in the OS store but not in uv's, so uv alone
fails. Same root cause as #3/#4 (why `ensure_ffmpeg.py` shells out to
`curl` instead of using `requests`) — it just reaches uv by a different
route, because `requirements.txt` pulls several deepbeepmeep wheels by
direct GitHub URL rather than from an index.

**Second, independent failure on the same step:** once certificates are
fixed, the next wheel failed with `operation timed out`. uv's default HTTP
timeout is 30 s, and GitHub release downloads measured **36 s for a 61 KB
file** on the affected connection — slow enough to fail the install with
nothing actually wrong.

**Fix — env on every uv invocation** (`install.js`, `update.js`,
`repair.js`, `rollback.js`, `torch.js`):

```js
env: { UV_SYSTEM_CERTS: "1", UV_HTTP_TIMEOUT: "180" }
```

`UV_SYSTEM_CERTS` makes uv use the OS store (schannel on Windows); it was
`UV_NATIVE_TLS` before uv 0.11 and that name still works but warns. Both
settings are harmless where they are not needed, hence unconditional.
`torch.js` needs the timeout most — those are multi-GB downloads.
Confirmed live: an update that had failed twice then completed cleanly.

### 9. Generation stalls at step 0 with the GPU idle — memory profile, not the model

**Symptom:** a generation that used to work sits at `0/N` for tens of
minutes. RAM ~99% used, GPU essentially idle, one CPU core busy, disk
quiet. Cancelling and picking a different model or text encoder does not
help.

**This is a memory-profile regression, and there is a one-line test for
it.** Check what mmgp preloads:

```
grep -o "Async loading plan for model 'transformer' : [^|]*" logs/api/start.js/<newest>
```

| Output | Meaning |
|---|---|
| `base size of 58.04 MB will be preloaded with a 369.18 MB async circular shuttle` | **streaming** (profile 4) — correct on a 32 GB machine |
| `13348.60 MB will be preloaded (base size of 58.04 MB + 72.0% of recurrent layers data)` | **preloading** — needs far more RAM than 32 GB with H3 |

**Root cause:** `wgp_config.json`'s **`video_profile`** (not the global
`profile`) drives video models. Value `3.5` — *"Profile 3+,
VeryLowRAM_HighVRAM: at least 32 GB of RAM and 24 GB of VRAM"* — maps at
`wgp.py:4231` to `mmgp_profile = 3` with `pinnedMemory = False`, and mmgp
profile 3 *"will try to load **entirely** a model in VRAM"*. Profile 4
loads *"only the needed parts"*.

On a machine at exactly 3.5's stated minimum (32 GB / 24 GB), a 20B
transformer plus a 32B text encoder does not fit, so it degrades into
paging and starves. Profile 4 — which upstream labels **(Recommended)** —
streams instead and works.

Maestro's auto-tune writes profile settings
(`services.auto_performance_applied: true` in `wgp_config.json`), so this
can change without the user touching it, e.g. across an upstream update.

**Fix:** set `video_profile` to `4` (Settings → Memory Profile → *"Profile
4, LowRAM_LowVRAM (Recommended)"*). Edit `wgp_config.json` only while
Maestro is stopped — it rewrites the file on exit. Also re-check **First
Block Cache**; it was off in the failing runs and on in every working one.

**Confirmed on an RX 7900 XTX / 32 GB / gfx1100 install:**

| | profile 3.5 (stalled) | profile 4 (working) |
|---|---|---|
| GPU compute | idle | **95.4%** |
| transformer preload | 13,348 MB | 58 MB |
| python RSS | 17.6 GB | 8.5 GB |
| packed rows | 27,783 (stalled) | **41,194 (running)** |

### 9b. How to read the counters (they mislead if you don't)

- **The Windows GPU compute counter does work for ROCm/HIP.**
  `Get-Counter '\GPU Engine(*engtype_Compute)\Utilization Percentage'`
  reads ~95% during a healthy generation, so a near-zero reading is real
  evidence the GPU is idle — not a broken counter. An earlier revision of
  this file claimed otherwise; that was wrong.
- **A high hard-fault rate is not automatically thrashing.** Profile 4
  streams weights through a ~369 MB circular shuttle, so ~30,000
  pages/sec (~124 MB/s) is *normal and healthy* — provided GPU compute
  stays saturated. Starvation looks like the opposite: a *low* fault rate
  **and** an idle GPU. Read the two together, never separately.
- **One CPU core pegged means nothing on its own.** That is what a
  PyTorch dispatch loop looks like from the host side; it appears in both
  the healthy and the starved case.

**Method note.** Diagnosing this took a long detour through the H3 text
encoder (NVFP4 vs GGUF) that was entirely a red herring — the failing run
was already on the *smaller* encoder. What settled it was diffing
`logs/api/start.js/*` between a known-good run and a failing one. Those
logs record the profile, preload plan, cache state, encoder, step count
and packed rows for every generation. **Diff the logs before theorising.**

### 10. Flat-grey output (a ~49 KB mp4 of solid RGB(128,128,128))

**Symptom:** an H3 generation runs to `[6/6] VAE Decoding`, the run
reports `Task 1 completed` / `New video saved`, and the saved mp4 is
**~48–51 KB** — every frame flat grey. Often (not always) an *"AMD
software detected that a driver timeout has occurred"* dialog appears near
the end. Same seed → **byte-identical** output. Survives reboots and a
Maestro rollback.

**What fixed it (2026-09-10, after a long wrong-tree chase — fp8 NaN,
v2.1.4 VAE rewrite, TDR, MIOpen cache all ruled out):
switching H3 Text Encoder from GGUF Q2_K to GGUF Q4_K_M.** Same seed /
640×640, real 9.1 MB video first try where Q2_K had produced grey.

**Root cause not fully pinned, but the working theory is: the 2-bit Q2_K
GGUF encoder is unreliable for H3 on this ROCm build.** Its embeddings are
marginal enough that some workloads tip into producing near-garbage
conditioning → the transformer emits a near-constant latent → the VAE
decodes it to mid-grey. This is consistent with #6b: **Q4_K_M is the
field-proven encoder and the `configure_amd_defaults.py` default; Q2_K was
never recommended, only offered as a low-RAM fallback.** A user who has
set Q2_K by hand and hits flat-grey output should be moved back to Q4_K_M
first, before anything else.

Not conclusively proven because the successful run also cut frames
175→141; but Q2_K had already failed at 640×640/175 where Q4_K_M/141
succeeded, and Q2_K failed across 640–768 resolution, so the encoder is
the far more likely variable. (An earlier revision of this note claimed
the Q2_K *file* was corrupt, mtime-matched to the failure onset — that was
a misread date; the file is the original 09-08 download, unchanged, and it
produced good videos earlier that same day. Deleting/redownloading it is
not the fix; switching encoder is.)

**Why it mimics a hardware/driver fault:**

| Grey-out clue | What it means |
|---|---|
| Same seed → byte-identical grey mp4 | Deterministic — bad conditioning, not a race |
| Survives reboot / "Roll back last update" | Not the driver, not the Maestro version |
| Dropping resolution / frames changes nothing | The encoder output is wrong regardless of size |
| The "AMD driver timeout" popup | **Secondary.** RAM starvation dragged the already-doomed decode out to 8–9 min, long enough to trip Windows' GPU watchdog. Annoying, not causal. |

**First move when H3 output goes flat grey: switch H3 Text Encoder to
GGUF Q4_K_M** (Advanced Settings). If it was already on Q4_K_M, then check
for a genuinely corrupt file — compare `stat` mtimes under
`Maestro/app/ckpts/minimax_h3/` against when it broke — and run **Repair**.

**Contributing factors that are real but were NOT the root cause:**

- **RAM starvation.** 32 GB is marginal for H3 at length — mmgp logs
  *"full requirements 19,987 MB while estimated available reservable RAM
  is 13,021 MB"* and runs permanently in partial-pinning mode. Seen at
  **0.4–0.9 GB free**. It makes every decode slower and is what lets the
  TDR watchdog fire, but good runs happened under the same pressure on the
  intact encoder file. Mitigate with a reboot before big jobs, `LLM
  Device: CPU`, fewer other apps, smaller generations.
- **Windows TDR watchdog.** Raising it hides the popup and lets a slow
  decode finish, but does not fix a corrupt-file grey-out. `TdrDelay` /
  `TdrDdiDelay` DWORDs under
  `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers` (needs admin +
  reboot; `TdrDelay=60` alone is not enough — `TdrDdiDelay` defaults to
  5 s). The dialog's *"Don't ask me again"* checkbox only suppresses the
  prompt; the GPU still resets.

**The Settings → System → "VAE Tiling" dropdown (`vae_config`) does
nothing for H3.** It only reaches VAEs implementing `get_VAE_tile_size`
(Wan, LTX, Hunyuan, Qwen, Flux); `wgp.py:7962`'s `hasattr` check is False
for `models/minimax_h3/video_vae.py`'s `AutoencoderKLMiniMaxH3`, so
`VAE_tile_size` stays `None` and `minimax_h3_main.py`'s `generate()` drops
the arg into `**_kwargs`. The H3 video VAE tiles itself, hard-on at a
fixed 256 px (`video_vae.py:663`). Do not tell a user to set "Aggressive
Tiling (Low VRAM)" for an H3 grey-out.

**Related but distinct failures seen in the same debugging session** (all
on `minimax_h3_ref2va`, 2nd+ generation in a long-lived server on a 32 GB
box — points at mmgp offload state not resetting cleanly between jobs):
`split_with_sizes ... sum to 64 ... got [4,4,4,4]` in
`models/ideogram4/qwen3_vl_transformers.py:947` (an upstream pre/post
spatial-merge grid bug — pure shape logic, no HIP in the trace, report
upstream), and a bogus `HIP OOM 9980 GiB` in the Q2_K embedding. Restart
the server between big H3 jobs.

### 11. ace_step crash: `ImportError: cannot import name 'group' from 'torch.distributed'`

**Symptom:** loading the ace_step TTS/music pipeline crashes at import
time with `ImportError: cannot import name 'group' from
'torch.distributed'`, coming out of `vector_quantize_pytorch`.

**Root cause — same family as issue #1, a call site the original FSDP
shadow didn't cover.** `vector_quantize_pytorch`'s
`lookup_free_quantization.py` (pulled in by ace_step) does
`from torch.distributed import nn as dist_nn` at module scope. That runs
the real `torch/distributed/nn/__init__.py`, which does
`from .functional import *`; `functional.py` does
`from torch.distributed import group, ReduceOp` at module scope. `group`
is only ever assigned inside `torch/distributed/__init__.py`'s
`if is_available():` block, and `is_available()` is `False` on this
ROCm-for-Windows nightly (no RCCL/GLOO) — so the import dies. The only
thing `dist_nn` is used for at that call site (`dist_nn.all_reduce`) is
itself guarded by a world-size>1 check, so it's dead code on Maestro's
single-GPU path — same shape as `shard_model()` in issue #1.

**Fix — `maestro_amd_preamble.py`, mirrors the FSDP shadow exactly.**
Pre-seeds `sys.modules["torch.distributed.nn"]` (+ `.functional`) with a
fake module before the real one can load, using the same
raises-if-actually-called placeholder shape as the FSDP stub. Never
imports `torch` itself, so it carries none of the issue #2 subprocess-storm
risk. `diagnose.py` grew a matching check next to `check_fsdp_shadow`.

**Verified end-to-end** on an RX 7900 XTX / Windows install running the
exact reported PyTorch build (`2.10.0a0+rocm7.10.0a20251120`):
reproduced the ImportError on the pre-fix venv; read the installed
`vector_quantize_pytorch` and `torch/distributed` source to confirm the
chain above; deployed the fix via `install_preamble.py` (the real
`install.js`/`update.js` entry point); post-fix, in a fresh interpreter,
`torch.distributed.nn`/`.functional` are shadowed automatically at
startup, `import vector_quantize_pytorch` (what ace_step does) succeeds,
the stub's `all_reduce` raises a clean `RuntimeError` if ever called
rather than crashing the interpreter, the existing FSDP shadow is
unaffected, and `diagnose.py` reports `[ok] torch.distributed.nn shadowed
by the wrapper preamble`.

### 12. Crash on every Start after Maestro v2.2.0: `KeyError: 'architecture'`

**Symptom:** `launch.py` dies importing `wgp.py`, before any UI or GPU code:
```
File ".../wgp.py", line 3483, in <module>
    load_model_definitions()
File ".../wgp.py", line 3452, in load_model_definitions
    model_def = init_model_def(model_type, model_def)
File ".../wgp.py", line 3414, in init_model_def
    base_model_type = get_base_model_type(model_type)
File ".../wgp.py", line 2877, in get_base_model_type
    return model_def["architecture"]
KeyError: 'architecture'
```
(Reported on Ubuntu / Radeon AI Pro 9700. The "RTX 50 / CUDA 13 ACTION
REQUIRED" lines printed just before it are upstream noise, unrelated.)

**Root cause — our own override file, broken by an upstream loader
change.** `configure_amd_defaults.py` used to write a two-key delta per H3
model to `finetunes/minimax_h3*.json`
(`{"_maestro_amd_managed": true, "model": {"minimax_h3_text_encoder_default": "gguf_q4_k_m"}}`).

- Up to **v2.1.6**, `load_model_definitions()` merged a same-named
  finetune over its default (`existing_model_def.update(model_def)`) and
  never re-ran `init_model_def`. The delta worked.
- From **v2.2.0** (upstream fix for Blizaine/Maestro#126, stale keys on
  live reload) it does `models_def[model_type] = <raw finetune dict>`, then
  `init_model_def()` → `get_base_model_type()` reads that raw dict → no
  `architecture` → crash. After init it **replaces** the default
  (`existing_model_def.clear(); existing_model_def.update(model_def)`), and
  the file's top-level keys become the model's UI `settings`.

**Why "just add `architecture`" is wrong.** It stops the crash, but under
replace semantics every H3 variant would lose its `name`, `URLs`
(checkpoint downloads), variant keys such as `minimax_h3_qkv_layout` and
all default settings. And `architecture` differs per variant
(`minimax_h3`, `minimax_h3_full`, `minimax_h3_voice_audio`, ...), so it
cannot be hardcoded.

**Fix — `configure_amd_defaults.py` writes a full copy** of the matching
`defaults/` file with only the encoder default changed, rebuilt from the
current default on every Start/Update (so it tracks upstream edits). The
ownership marker moved inside `"model"` — at top level it would now leak
into `settings/<model>_settings.json`; top-level markers are still
recognised so old files get rewritten. Managed copies whose default was
removed upstream are deleted. **Self-healing:** `start.js` runs the script
right before `launch.py`, so an affected user only needs the wrapper
update and one Start. `diagnose.py` fails on any `finetunes/*.json`
Maestro would refuse to load, using the script's own predicates.

**Hardening — so the next upstream loader change can't take Start down.**
The loader also `raise`s on an *unparseable* finetune, and on any file
lacking `model`/`model.architecture`, whoever wrote it. The first
shipped fix still had three ways back into a crash: a managed file
corrupted mid-write was mistaken for a user file and skipped forever;
a default that stopped carrying `architecture` left the old override in
place; and a broken user finetune crashed Start like before. Now:

- **Quarantine pass first.** Every `finetunes/*.json` the installed loader
  would crash on is removed if it is ours (it gets rebuilt), otherwise
  renamed to `<name>.json.disabled` — never deleted. Diagnose warns about
  every `.disabled` file with the restore step.
- **User files are only renamed when the installed `wgp.py` really rejects
  them.** `loader_traits()` reads its source: *strict* (indexes
  `json_def["model"]` / `model_def["architecture"]`) and *replaces*
  (`existing_model_def.clear()`, v2.2.0+). On a merge loader (≤ v2.1.x) a
  user's partial override of a same-named default loads fine, so it stays;
  on a loader the check doesn't recognise, parseable user files are never
  touched.
- **Stock upstream is the fallback.** If a default is unreadable or has no
  `model.architecture`, our copy is removed rather than kept stale.
- **Atomic writes** (`<name>.json.tmp` + `os.replace`; the tmp suffix
  avoids the loader's `*.json` glob). Each pass is isolated; exit is
  always 0.

**Verified** with `test_amd_defaults.py`, including its loader contract,
which lifts upstream's real `load_model_definitions()` out of `wgp.py` by
AST and runs it on the wrapper's output — against v2.1.5 (merge), v2.2.0
and v2.2.2 (replace), seeded with a live install's nine old delta files:
they raise `KeyError: 'architecture'` on 2.2.0/2.2.2 and load on 2.1.5;
after one run of the script all nine H3 variants load on all three with
name/URLs/architecture intact and the encoder on `gguf_q4_k_m`.
**After an upstream update, run the contract test against the new
`wgp.py` first** — if it fails or skips ("loader functions renamed"), the
loader changed and this section needs revisiting.


## Do not

- Do not vendor upstream Maestro source files into this repo (that was
  the abandoned approach — remnants in `temp/`, safe to delete when
  convenient).
- Do not add NVIDIA / CUDA / Sol / RTX branches. Sibling project.
- Do not write partial (delta) model definitions to `finetunes/`. Since
  Maestro v2.2.0 a same-named finetune replaces its default outright —
  always write a full copy of the matching `defaults/` file. And never
  delete a finetune the wrapper did not write — rename it aside. See
  "Known runtime issues" #12.
- Do not use `--depth 1` for the Maestro clone — the extra weight is
  negligible and it keeps `git pull` trivially correct across upstream
  branch rewrites.
- Do not swap `git reset --hard` for `git pull` in `update.js` without
  thinking — `git pull` fails on divergent local edits and leaves the
  user stuck. `reset --hard origin/HEAD` is the resilient shape.
- Do not gate `install.js` on `requires: { bundle: "ai" }` — that
  assumes NVIDIA-oriented deps.
- Do not use Python's `requests` (or anything relying on `certifi`'s
  bundled CA list) for downloads in this repo's helper scripts — shell out
  to `curl` instead. See "Known runtime issues" #3/#4 for the observed
  failure.
- Do not, in `sitecustomize.py` or anywhere else, `import
  torch.distributed.fsdp` (or anything under it, or anything that
  transitively runs its `__init__.py`) — see "Known runtime issues" #2.
  The current fix writes to `sys.modules` unconditionally without
  probing; that's the only verified-safe shape.
