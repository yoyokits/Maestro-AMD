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
├── install.js              ← clone + build (AMD-guarded)
├── torch.js                ← ROCm wheel install, per gfx target
├── start.js                ← daemon: python launch.py
├── update.js               ← git pull + refresh deps
├── start_latest.js         ← "Update & Start" one-click
├── reset.js                ← rm -rf Maestro/
├── launcher_profile.js     ← AMD GPU detection + runtime profile
├── sitecustomize.py        ← copied into venv site-packages, see "Known runtime issues" #1
├── install_sitecustomize.py← the copier itself, called from install.js/update.js
├── ensure_ffmpeg.py        ← provisions ffmpeg/ffprobe, see #3/#4
├── test_gpu_detection.js   ← unit tests for launcher_profile/torch.js detection logic
├── CLAUDE.md               ← this file
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
(`install_sitecustomize.py`, `ensure_ffmpeg.py`) with `path: "."`. That
created a stray `env-amd` at the **app root**, and `sitecustomize.py` +
`ffmpeg`/`ffprobe` landed there — while the real runtime venv at
`Maestro/app/env-amd` (used by `torch.js` and `start.js`, which correctly
use `path: "Maestro/app"`) never got the FSDP shadow. Result: `launch.py`
crashed on the `torch.distributed.fsdp` import — the exact symptom of
Known runtime issue #1, despite the fix being "installed".

**Fix.** Helper steps now use `path: runtime.path` and invoke the helper
by absolute path (`python "${__dirname}/install_sitecustomize.py"`), so
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

1. `git -C Maestro fetch origin && git -C Maestro reset --hard origin/HEAD`
   — matches upstream tracked files exactly. **Wipes any manual edits**
   inside `Maestro/`. Untracked user data (models, outputs, config) is
   preserved because `git reset` only touches tracked files.
2. Re-runs `uv pip install -r requirements.txt` in the venv.
3. Runs `torch.js` **only if** the runtime marker
   (`Maestro/app/env-amd/.maestro_amd_v1.installed`) is missing. Delete
   that marker to force a full ROCm wheel reinstall on the next update.
4. Self-heals the `seedvc` component if the user deleted it.
5. Rebuilds the React UI (`npm install && npm run build`).

For **easy/automatic updating** the wrapper exposes two shapes in the
Pinokio menu:

- **Update** — just runs `update.js`.
- **Update & Start** — runs `start_latest.js`, which chains `update.js`
  → `start.js`. One click, always latest. This is the recommended
  everyday launcher.

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

For the async-export files (install/torch/start/update),
`node --check` catches parse errors but not runtime issues in the
returned config object. Test the runtime path via Pinokio.

`sitecustomize.py`, `install_sitecustomize.py`, and `ensure_ffmpeg.py`
are plain Python — check with
`python -m py_compile sitecustomize.py install_sitecustomize.py ensure_ffmpeg.py`
using the same `env-amd` venv install.js/update.js invoke them with, then
actually run them against a real `Maestro/` clone (all are idempotent and
safe to re-run) before trusting a change to any. For `sitecustomize.py`
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

**Fix — automated, `sitecustomize.py` + `install_sitecustomize.py`.** Zero
modification to Maestro's tree. `install_sitecustomize.py` (invoked by
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
`sitecustomize.py`, `install_sitecustomize.py`, and their call sites in
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

## Do not

- Do not vendor upstream Maestro source files into this repo (that was
  the abandoned approach — remnants in `temp/`, safe to delete when
  convenient).
- Do not add NVIDIA / CUDA / Sol / RTX branches. Sibling project.
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
