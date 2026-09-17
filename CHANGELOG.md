# Changelog

App versions for the Maestro AMD **wrapper**. Tagged `vX.Y.Z` on this
repo.

Not to be confused with `version: "8.0"` in `pinokio.js` — that is the
**Pinokio script schema version**, which declares which Pinokio features
the script may use, and tracks Pinokio releases rather than this project.
Do not repurpose it for app versioning; lowering it removes access to
script features (Pinokio: *"applications with SCRIPT VERSION 2.0 or
higher are only supported on Pinokio 2.0 or higher"*).

The version of upstream Maestro you are running is independent of this —
it is whatever `Maestro/` is checked out at. **Diagnose** prints both.

## Unreleased

### Fixed

- **RDNA 2 cards other than the RX 6800/6900 crashed the server on Linux.**
  PyTorch's `rocm7.2` wheels carry no kernels for gfx1031/gfx1032/gfx1034
  (RX 6600/6650/6700/6750), so the GPU reported itself as available and
  then segfaulted on the first real kernel launch — reported as a Director
  song upload dying with "failed to fetch" and a greyed-out Start button,
  because the segfault takes the whole backend down with it (issue #3,
  brcisna). Start and Diagnose now set `HSA_OVERRIDE_GFX_VERSION=10.3.0`
  for RDNA 2 on Linux. Windows was never affected.

### Added

- **Diagnose reports whether PyTorch actually ships kernels for your GPU**,
  comparing the device's gfx target against `torch.cuda.get_arch_list()`.
  This is the failure above, generalized — it will catch the next wheel
  that drops a target rather than letting it segfault.
- `MAESTRO_AMD_AUDIO_SEPARATOR_DEVICE=cpu` (via `user_env.json`) runs the
  RoFormer vocal extractor on the CPU, for a GPU that cannot run that model
  at all. Off by default: on a working GPU it is dramatically slower
  (measured: 19 s vs. no progress in 11 minutes for a 60 s clip).

## v0.2.0

Hardening release: GPU detection, venv correctness, diagnostics, and
recovery actions.

### Fixed

- **Strix Halo installs failed silently.** Pinokio's GPU detection
  returns `gpu_target: null` for APU iGPU names ending in "Graphics", so
  every wheel-install step skipped, the runtime marker was never written,
  and Start reported "AMD ROCm runtime not installed". `resolveGpuTarget`
  now resolves the target in JS from the raw GPU model strings.
  (thomas9120)
- **Helper scripts provisioned the wrong virtualenv.** Pinokio resolves
  `venv` relative to each step's `path`, so helpers running with
  `path: "."` created a stray `env-amd` at the app root — the real
  runtime venv never received the FSDP preamble or ffmpeg, and `launch.py`
  crashed on the `torch.distributed.fsdp` import. Every venv-touching
  step now uses `path: runtime.path`. (thomas9120)
- **A CPU-only PyTorch could silently replace the ROCm build**, giving
  `AssertionError: Torch not compiled with CUDA enabled` roughly forty
  frames deep in `t5.py`. ROCm wheels now install before
  `requirements.txt`; `verify_rocm_torch.py` detects a wrong build and
  Update reinstalls automatically; Start refuses with an explanation
  instead of the traceback.
- **Update could fail on `uv` TLS and timeouts.** `uv` validates against
  its own bundled CA roots, so it failed with `invalid peer certificate:
  UnknownIssuer` behind TLS inspection while `git` and `curl` succeeded;
  and its 30 s default timeout was too short for GitHub release wheels
  (measured: 36 s for 61 KB). Now `UV_SYSTEM_CERTS=1` and
  `UV_HTTP_TIMEOUT=180` on every `uv` call.
- **Unsupported GPUs burned a full install before failing.** `install.js`
  now rejects them before the clone.
- **The menu offered "Start" after a failed install.** It gates on the
  runtime marker rather than the venv directory.
- **MiniMax H3 defaulted to an NVIDIA-only text encoder** (NVFP4 AWQ,
  labelled "Recommended"), which has hung entire machines on AMD. Now
  overridden to GGUF Q4_K_M via `finetunes/`, which is git-ignored
  upstream and survives `git reset --hard`. A stale `attention_mode`
  naming an uninstalled backend is reset to `auto`.

### Added

- **Diagnose** — a health report covering install layout, PyTorch build,
  GPU, ffmpeg, the FSDP preamble, AMD setting overrides, Triton, and
  which SDPA attention backends are actually usable. Attach it to bug
  reports.
- **Repair** — rebuilds the environment while keeping models, LoRAs and
  outputs. Previously the only recovery was Reset, which deletes them.
- **Roll back last update** — returns to the pre-update Maestro revision.
- **Install Inpaint Support** — SAM 3.1 in an isolated environment.
- `user_env.json` for per-machine environment overrides.
- `PERFORMANCE.md`: what the NVIDIA path installs that this wrapper
  cannot, why, and which gaps are closable.

### Changed

- The FSDP preamble moved from `sitecustomize.py` to a `.pth` hook.
  site-packages holds only one `sitecustomize`, so any dependency
  shipping one would have silently disabled the fix. Existing installs
  migrate automatically.
- Update skips the React UI rebuild when `ui/` has not changed.
- `ensure_ffmpeg.py`: path-traversal-safe extraction, a clear error when
  `curl` is missing, and a post-install verification run.

### Known limitations

- Not yet exercised end-to-end through Pinokio's UI on a fresh install.
  `repair.js`, `rollback.js` and `sam_install.js` are untested in
  practice.
- The AOTriton attention fix (`CLAUDE.md` #5) is confirmed only on
  gfx1100. Coverage for RDNA 2 and the gfx115x APUs is doubtful — run
  **Diagnose** on that hardware to find out.

## v0.1.0

Initial AMD ROCm wrapper around [Blizaine/Maestro](https://github.com/Blizaine/Maestro):
per-gfx-target ROCm wheel installation for RDNA 2–4 and Strix/Krackan
APUs, the FSDP import-crash workaround, ffmpeg/ffprobe provisioning, the
AOTriton SDPA fix for the O(n²) attention OOM, and the
Update / Update & Start / Reset menu.
