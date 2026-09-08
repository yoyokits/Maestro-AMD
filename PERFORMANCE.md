# AMD performance: what's missing vs CUDA, and what we can do about it

Companion to `CLAUDE.md`. That file documents what the wrapper *does*;
this one documents what the wrapper **can't do yet** on AMD, why, and
which gaps are actually closable from here.

Compiled September 2026 from upstream Maestro (`torch.js`,
`app/requirements.txt`, `app/shared/attention.py`,
`app/services/optional_acceleration.py`,
`app/scripts/install_optional_cuda_acceleration.py`,
`app/scripts/install_gguf_kernels.py`), AMD's own release notes and
support matrices, and the ROCm/TheRock issue trackers. Re-check when
bumping the ROCm wheel or when upstream changes its accel stack.

Every claim below is tagged:
**[verified]** — checked directly against a live index, file, or AMD doc
in this repo's own research.
**[reported]** — from an upstream issue, vendor blog, or benchmark we did
not reproduce. Treat as a lead, not a fact.

---

## 0. Read this first: AOTriton coverage is narrower than our target list

`CLAUDE.md` "Known runtime issues" #5 documents the OOM fix —
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` in `start.js`, which turns on
AOTriton's Flash / memory-efficient SDPA kernels instead of the O(n²)
Math fallback. That fix was verified on **gfx1100 (RX 7900 XTX)**.

**It does not follow that it works on every target `torch.js` installs
for.** AOTriton ships *precompiled* kernels for a specific target list.
Where a target isn't in that list, the env var is a no-op and SDPA still
falls back to Math — i.e. the exact 92 GiB-style OOM from #5, with no
wrapper-side fix available.

| Our target | AOTriton kernels? | Consequence |
|---|---|---|
| gfx1100/1101/1102 (RDNA 3 dGPU) | Yes **[verified]** — #5 fix confirmed on gfx1100 | Working |
| gfx1200/1201 (RDNA 4) | Listed as a build target **[reported]** | Expected working, **unverified on hardware** |
| gfx1150/1151/1152/1153 (APUs) | **Contested.** AOTriton's target range is quoted as "gfx1100 through gfx1153" **[reported]**, but ROCm/ROCm#5404 reports `AOTriton not available: No module named 'pyaotriton'` on gfx1151 with SDPA falling back to `math` and "memory requirements exceeding 100GB for 2-3 seconds of video" **[reported]** | **Likely still OOMs.** Version-dependent — depends on whether the specific nightly wheel bundles gfx115x kernels. |
| gfx1030/1031/1032/1034 (RDNA 2) | **No.** AOTriton's documented range starts at gfx1100 **[reported]** | **Expected to OOM on real generation lengths.** No env-var fix exists. |

Two things follow:

1. **This is the single highest-priority item in this document.** It is
   not a "make it faster" gap, it is a "does it work at all" gap, and it
   affects targets `torch.js` advertises support for. Everything in §2 is
   optimization; this is correctness.
2. **It must be verified on hardware before we act.** We have one
   confirmed-good target (gfx1100). The rest is inference from issue
   trackers. See §4 for the check.

Candidate mitigation for the APUs, from ROCm/TheRock#1364 **[reported]**:
install the **`gfx110X-dgpu`** wheels and set
`HSA_OVERRIDE_GFX_VERSION=11.0.0`, which makes gfx1151 use gfx1100's
kernels. Reported to make Flash Attention work there. This is a real
candidate for a `when`-gated branch in `torch.js` + `start.js` — but it
overrides AMD's own per-target wheel split, so it needs testing on an
actual Strix Halo box before it goes anywhere near `main`.

For RDNA 2 there is no equivalent lever. If the OOM is confirmed there,
the honest options are: document the limitation, cap generation length,
or drop the target from the support table.

---

## 1. What we are NOT missing

Do not spend effort here.

- **`mmgp==3.7.12`** — deepbeepmeep's offload / partial-pinning engine,
  and by a wide margin the most important performance component in the
  stack. Plain Python, in `requirements.txt`, installs and runs on AMD
  unchanged. **[verified]**
- **`accelerate==1.12.0`** — same story.
- **Distillation LoRAs** (CausVid, self-forcing, LightX2V LoRAs) — these
  are *weights*, vendor-neutral, and work. Only the matching CUDA
  *kernels* are missing (§2).
- **`xformers`** — CUDA-only and absent, but functionally redundant with
  SDPA/AOTriton. Its absence costs approximately nothing.

The denoise inner loop on a gfx1100-class card is in reasonable shape.
What's missing is the tier layered on top.

---

## 2. What the NVIDIA path installs that we don't

Exact specs from upstream `torch.js` (Sol / CUDA 13 path). **[verified]**

| Component | NVIDIA spec | Accelerates | Impact on AMD |
|---|---|---|---|
| **triton** | `triton-windows==3.6.0.post25` (Win), `triton>=3.6,<3.7` (Linux) | keystone: Sage, FA2-triton, `torch.compile`, fast `flash-linear-attention` | **High** — blocks most of this table |
| **torch.compile** | `--compile`; Sol default is `--compile --attention sage2` | whole-graph fusion | **High potential**, needs only Triton + the flag |
| **GGUF dequant kernels** | `llamacpp_gguf_cuda` wheel | on-the-fly dequant of `*_gguf_*` variants; CPU fallback otherwise | **High in one specific case** — see §3.3 |
| **FP8 / NVFP4 matmul** | Ada/Hopper FP8, Blackwell NVFP4 | quantized text encoders + transformers | **High on RDNA 3** — see §3.4 |
| **SageAttention v1** | optional wheel, Triton-gated, SM70+ | INT8 quantized attention | Medium, **uncertain payoff** — see §3.5 |
| **SageAttention v2/v3** | `sageattention-2.2.0+cu130`; v3 Blackwell | FP8/INT8 attention, SM80+/SM100+ | Medium. No ROCm build. `attention.py`'s default chain is `sage2 → sage → sdpa`; on AMD it always lands on `sdpa`. |
| **FlashAttention-2** | `flash_attn-2.8.3+cu130torch2.10` prebuilt wheel | fused attention | Medium — CK-Tile is Linux-only; Triton-AMD backend needs a source build |
| **radial / block-sparse** | `spas_sage_attn.block_sparse_sage2_attn_cuda` | sparse attention for **long video** | Medium for long-form; `--attention radial` is dead on AMD |
| **nunchaku** | `nunchaku-1.2.1+cu13.0torch2.10` | SVDQuant 4-bit transformers (FLUX / Qwen-Image class) | Medium. CUDA-only, no ROCm port exists. |
| **lightx2v_kernel** | `lightx2v_kernel-0.0.2+torch2.10` | distilled / few-step video kernels | Medium. CUDA-only. |
| **cuDNN vs MIOpen** | cuDNN (mature) | convolutions — VAE encode/decode | Low–medium. MIOpen is unstable; `start.js` already sets `MIOPEN_FIND_MODE=FAST` + `MIOPEN_DISABLE_CACHE`. |
| **onnxruntime GPU EP** | `onnxruntime-gpu==1.22.0` (CUDA EP) | preprocessors: depth, pose, face-detect, `rembg`, `insightface` | Low–medium. CUDA EP won't engage; falls to CPU. Preproc, not the denoise loop. |

`flash-linear-attention==0.4.1` is in `requirements.txt` so it *installs*
on AMD, but wants Triton to be fast. Without it, slow path. **[verified]**

---

## 3. What can actually be implemented here

Ranked by (leverage ÷ risk). Read §0 first — none of this matters on a
target that can't do fast attention at all.

### 3.1 Triton — the keystone, and it splits by platform

The decisive fact, and it is **not** symmetric:

- **Linux:** `pytorch-triton-rocm` is present in
  `https://download.pytorch.org/whl/rocm7.2/` **[verified — checked the
  index directly]**. It is a dependency of the ROCm torch wheel, so
  **Linux AMD users already have Triton installed today.** `torch.compile`
  and the Triton-backend attention libraries are reachable on Linux with
  no new wheel at all.
- **Windows:** AMD's `v2-staging/gfx110X-dgpu` index contains only
  `torch`, `torchvision`, `torchaudio`, `rocm-sdk-*`, and their plain
  Python deps — **no Triton of any kind** **[verified — enumerated the
  index]**. Triton on Windows must come from the third-party
  `triton-windows` fork (latest `3.8.0.post28`, 29 Aug 2026
  **[verified via PyPI]**), whose AMD support is explicitly experimental
  and covers **gfx1100+ only** **[reported]**.

**Implication for `torch.js`:** the Linux branch could enable Triton-gated
features essentially for free. The Windows branch cannot, and any Windows
Triton step must be `when`-gated to RDNA 3/4 dGPU targets, leaving RDNA 2
and every APU on the current path.

This asymmetry is the most useful planning fact in this document. If we
want to demonstrate a Triton-based win at low risk, **do it on Linux
first.**

### 3.2 `torch.compile` — best effort-to-payoff ratio

No new wheel on Linux (Triton is already there). Just pass `--compile`
to `python launch.py` in `start.js`. Biggest potential general speedup in
the whole document.

Caveats: historically flaky on the ROCm-Windows nightly **[reported]**;
adds a multi-minute first-run compile; can fail hard on graph breaks.
Gate it, don't default it, and measure.

### 3.3 GGUF dequant kernels — small change, concrete user-facing win

Upstream's `install_gguf_kernels.py` installs `llamacpp_gguf_cuda`, is
CUDA-only, and degrades to a **CPU dequant fallback** for `*_gguf_*`
model variants **[verified]**.

This bites us specifically: `CLAUDE.md` #6 routes MiniMax H3 users onto
GGUF to avoid the NVFP4 hang — straight into this gap.

**Status: unquantified, and an earlier claim here was retracted.** A
revision of this file asserted, from one stalled generation, that GGUF
made H3 unusable on AMD. The install's own logs then showed **nine**
successful H3 generations with GGUF Q4_K_M over two days at 6 steps, up
to 52,781 packed rows. The stalled runs were all on the *experimental
4-step fused-turbo* variant, which is the variable that actually
correlated. See CLAUDE.md #6b.

The CPU-dequant gap is still real in principle -- `llamacpp_gguf_cuda` is
CUDA-only -- but note that `shared/qtypes/gguf.py` carries torch
implementations for Q4_K and Q2_K (`_DEQUANTIZE_FUNCTIONS`), so those
variants do *not* take the numpy CPU path. Its practical cost on AMD is
**unmeasured**. Do not quote a figure for it without one.

This is the highest-value upstream ask in this document: llama.cpp itself
has a working HIP backend, so the kernels exist — what is missing is a
packaged ROCm wheel equivalent to `llamacpp_gguf_cuda`.

Encouragingly, llama.cpp itself has a working HIP/ROCm backend on
Windows (`ggml-hip.dll`) **[reported]**, so equivalent kernels exist —
what's missing is a packaged wheel. Realistically this is **upstream
work** (ask deepbeepmeep for a HIP build) rather than something the
wrapper can solve, but it's worth raising because the fix benefits every
AMD user of every GGUF variant.

### 3.4 FP8 — RDNA 4 only, and worth gating for

AMD's own Windows support matrix states FP8 is **"Supported only on RDNA4
GPUs"** **[verified]**. RDNA 3 has no FP8 matmul at all, which is the
hardware root of `CLAUDE.md` #6 (NVFP4 text encoder → silent CPU
fallback → OS-level hang).

So: any future FP8 quantization path must be gated to gfx1200/1201. And
the #6 guidance could reasonably become target-aware — RDNA 4 users may
have options RDNA 3 users don't.

### 3.5 Sage / FA2-triton — lowest priority, do last

Both need Triton (§3.1), so on Windows they inherit the gfx1100+
restriction. Two reasons to keep expectations low:

- AMD's own ComfyUI attention-backends benchmark found **SageAttention
  performed *worse* than PyTorch SDPA** on the tested RDNA 3 config, and
  describes it as experimental / not officially supported **[reported]**.
- `EmbeddedLLM/SageAttention-rocm` — the ROCm fork — advertises 2.1×
  vs FA2, but its published benchmarks are all **NVIDIA datacenter parts
  (A100/4090/L40)**, with **no RDNA 3 consumer numbers**, requirements
  still written in CUDA terms, source build only, and last substantive
  update **2024-11-21** **[verified]**. Do not treat its headline number
  as applicable here.
- SageAttention also has model-specific breakage — e.g. MiniMax H3 fails
  with `sageattn_varlen only supports head_dim [64, 128]` **[reported]**.

Upstream Wan2GP does document a working Windows AMD recipe
(`triton-windows`, `sageattention<2` with the `.post26` wheel, FA2 via
`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) **[reported]**, so it's *possible*.
It's just not obviously *worth it*, and it drags in a clang-cl / ninja /
ROCm-SDK source-build toolchain that fights the wrapper's minimal shape.

**One more reason for caution:** Triton compilation invokes
`offload-arch` for HIP arch detection — the same binary implicated in the
runaway subprocess storm in `CLAUDE.md` "Known runtime issues" #2. Prime
the watchdog before the first Triton compile on Windows.

---

## 4. Verification recipes

Nothing above should be acted on from reading alone. Run these in the
`env-amd` venv on the target hardware.

**Does this target have working fast attention?** (the §0 question)

```python
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
q = torch.randn(1, 8, 4096, 64, device="cuda", dtype=torch.float16)
for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q, q, q)
        print(be, "OK")
    except Exception as e:
        print(be, "FAILED:", e)
```

Run it with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` set, exactly as
`start.js` does. If both fail, this target is on the Math fallback and
will OOM on real generation lengths.

**Is Triton present and functional?**

```
python -c "import triton; print(triton.__version__)"
```

**Did an attention change actually help?** Don't trust micro-benchmarks.
Run the same real generation (same model, seed, resolution, frame count)
before and after, and compare wall-clock plus peak VRAM
(`torch.cuda.max_memory_allocated()`).

---

## 5. Running the experiments — `user_env.json`

`start.js` hardcodes five environment variables. To A/B any of them
without editing a tracked file, create **`user_env.json`** next to
`start.js` — a flat object of string values, git-ignored, applied over
the defaults at launch. A `null` value unsets a default entirely:

```json
{
  "HSA_ENABLE_SDMA": null,
  "MIOPEN_FIND_MODE": "NORMAL"
}
```

`start.js` logs which overrides it applied, and malformed JSON is
reported and ignored rather than blocking startup. "Diagnose" reports
whether the file is active.

Three experiments worth running, in order of expected value:

1. **`HSA_ENABLE_SDMA: null`** — this is the only one of the five
   defaults that isn't obviously a win. It was carried over from
   wan2gp-amd for parity. Disabling the DMA engines pushes host<->device
   copies onto other paths, and mmgp's entire offload strategy is heavy
   host<->device shuttling — so on a driver without the SDMA bug it may
   be *costing* throughput on exactly the hot path. Free to test.
2. **`torch.compile`** — on Linux only, where Triton is already
   installed (§3.1). Not wired to an env var; add `--compile` to the
   `python launch.py` line in `start.js` locally and measure.
3. **`PYTORCH_HIP_ALLOC_CONF`** — nothing to test on Windows: PyTorch
   reports `expandable_segments not supported on this platform` at the
   first allocation there (confirmed on a live RX 7900 XTX, torch
   2.10.0a0+rocm7.10.0). It is a no-op, not a tuning knob, on that
   platform. Still meaningful on Linux ROCm.
4. **`MIOPEN_FIND_MODE: "NORMAL"`** — `FAST` skips exhaustive kernel
   autotuning. `NORMAL` costs a slow first run per shape but may pick
   better convolution kernels for repeated work (VAE encode/decode).

Measure with a real generation — same model, seed, resolution and frame
count — comparing wall-clock and `torch.cuda.max_memory_allocated()`.
Micro-benchmarks do not predict this workload.

## 6. Ground rules for anything added from this document

- **Never change the default for targets that can't use it.** Every step
  from §3 is `when`-gated or an opt-in menu entry. RDNA 2 and APU users
  must keep working exactly as they do now.
- **Linux and Windows are different problems.** Triton is free on Linux
  and third-party+experimental on Windows (§3.1). Don't write one branch
  for both.
- **Measure before committing.** AMD's own benchmarks contradict several
  of the upstream performance claims. A change that doesn't show a clear
  win on real hardware doesn't go in.
- **Don't vendor upstream source to get any of this.** Same rule as
  `CLAUDE.md`: env vars, launch flags, and venv-local installs only. If
  it needs a Maestro code change, send it upstream.

---

## 7. Wrapper changes already made

Implemented September 2026 off the back of this analysis. All verified by
executing the Pinokio config builders in Node against simulated GPU
targets, and the Python helpers against real venvs and git repos — but
**none of it has been driven through Pinokio itself**, which is the only
way to exercise the real install path (`CLAUDE.md`, "Development /
testing changes"). Treat as reviewed, not field-tested.

**Diagnostics**

- **`diagnose.py` + `diagnose.js`, wired to a "Diagnose" menu entry.**
  Reports host, install layout, upstream revision, runtime env,
  ffmpeg/ffprobe, the FSDP preamble, the torch build, and Triton — and
  runs the §0 attention probe under the same environment `start.js` uses,
  so it answers "does this GPU get fast attention?" directly. Output is
  ASCII-only so it survives being pasted into an issue. Exits 0 always
  (it's a report, not a gate) and never touches
  `torch.distributed.fsdp` (`CLAUDE.md` #2).

**Correctness**

- **NVIDIA-only defaults are now overridden on AMD**
  (`configure_amd_defaults.py`). The MiniMax H3 text encoder defaulted to
  **NVFP4 AWQ**, an NVIDIA-only format implicated in a whole-OS hang
  (`CLAUDE.md` #6) -- it now defaults to **GGUF Q4_K_M**, which also
  redirects the download to the file the user can actually run. A stale
  `attention_mode` naming an uninstalled backend (Sage/Flash/xformers) is
  reset to `"auto"`. Applied via `app/finetunes/*.json`, which is
  git-ignored upstream and merges over `defaults/` -- so no upstream file
  is touched and the override survives `git reset --hard`.
- **A CPU-only torch can no longer silently break an install.** Reported
  by a user: `PyTorch 2.14.0+cpu`, then `AssertionError: Torch not
  compiled with CUDA enabled` forty frames deep in `t5.py`. Three layers
  now cover it -- ROCm wheels install before `requirements.txt`;
  `verify_rocm_torch.py` flags a wrong build after every requirements
  pass and `update.js` re-runs `torch.js` on that flag (so Update
  self-heals); and `--preflight` in `start.js` shows a readable dialog
  naming the cause instead of the traceback. `CLAUDE.md` #7 has the full
  write-up. The preflight is metadata-only and only acts on version
  strings that *prove* the build is wrong, so it cannot block a working
  install.
- **Fail-fast on unsupported GPUs.** `install.js` rejects an unsupported
  gfx target *before* the clone, venv and multi-GB requirements install,
  instead of dead-ending inside `torch.js` afterwards and leaving a venv
  with no torch in it. Windows only — the Linux stable index isn't
  gfx-gated.
- **The menu now agrees with `start.js`.** `pinokio.js` gated "installed"
  on the venv *directory*, which exists after a failed install too, so it
  offered Start and `start.js` immediately refused. It now tests the ROCm
  runtime marker, and offers Repair + Diagnose for a half-finished
  install.
- **Wrapper state moved out of the upstream clone** to `.maestro_state/`.
  It was under `Maestro/`, where `repair.js`'s own `git clean -fdx` would
  have deleted the rollback point at exactly the moment it was needed.
  Caught by the test harness, not by review.

**Install efficiency**

- **ROCm wheels install before `requirements.txt`.** Nothing pins torch,
  but torchcodec/accelerate/peft/timm/open_clip_torch all depend on it,
  so resolving them first pulled the PyPI torch purely to have `torch.js`
  overwrite it. Measured waste (torch 2.14, cp311): **~0.12 GB on
  Windows, ~1.6 GB on Linux** — the Linux figure is mostly
  `nvidia-cudnn`/`cusparselt`/`nccl`/`nvshmem`, which torch declares
  `platform_system == "Linux"` only. Smaller than assumed on Windows;
  still worth not doing. `verify_rocm_torch.py` is the safety net: it
  checks the installed build (metadata first, importing torch only when
  the version string is ambiguous) and flags a reinstall if the resolver
  replaced it.
- **The UI rebuild is now conditional.** `npm install && npm run build`
  ran on every Update, including the everyday "Update & Start" path.
  `update_state.py ui-check` compares the built revision against HEAD and
  skips when nothing under `ui/` changed. Fails open.
- **`RUNTIME_MARKER_VERSION`** in `launcher_profile.js` makes the
  marker-name bump a one-line change, so a future wheel-spec change
  actually reaches existing installs instead of being silently skipped.
  Deliberately *not* bumped for these changes — the wheels didn't change,
  and bumping would force everyone through a pointless re-download.

**Recovery**

- **`repair.js` — "Repair (keep models)".** Rebuilds the environment via
  `git clean -fdx` with the user-data paths excluded. Reset deletes tens
  of GB of checkpoints; this was previously the only way to fix a broken
  install.
- **`rollback.js` — "Roll back last update".** `update.js` records the
  pre-update SHA, so a bad upstream commit no longer leaves
  models-destroying Reset as the only escape. The menu entry only appears
  when a rollback point exists.

**Features**

- **`sam_install.js` — Inpaint / SAM 3.1 support**, ported from upstream.
  Isolated Python 3.12 conda env, so it cannot affect the ROCm venv; the
  only AMD-specific change is resolving the torch index from
  `launcher_profile.js` instead of upstream's hardcoded CUDA 12.8.
  Experimental on AMD — SAM 3.1 is ordinary PyTorch with no CUDA-only
  kernels, but it is not on the verified path.
- **`user_env.json`** (§5) — per-machine env overrides for `start.js`.

**Hygiene**

- **One source of truth for GPU routing.** `launcher_profile.js` now owns
  the gfx patterns *and* the ROCm wheel index table; `torch.js` builds its
  `when`-gated steps from them and `sam_install.js` reads the same table.
  Previously the regexes were duplicated between the two files — the
  exact drift `CLAUDE.md` warns about — and the `isAmdRdna*` helpers were
  exported but never called by anything.
- **`sitecustomize.py` -> `_maestro_amd_preamble.pth` +
  `maestro_amd_preamble.py`.** site-packages has room for exactly one
  `sitecustomize` module, so any dependency shipping one would have
  silently clobbered the FSDP fix. `.pth` hooks compose, and run slightly
  earlier. `install_preamble.py` migrates existing installs and leaves a
  third-party `sitecustomize.py` alone. Verified in a real venv: the hook
  fires at startup, satisfies Maestro's import surface, and **never
  imports torch** — the property that matters for `CLAUDE.md` #2.
- **`ensure_ffmpeg.py`** — path-traversal-safe extraction, an explicit
  error when `curl` is missing, and a post-install `-version` check so a
  broken binary is caught at install time rather than mid-generation.

### Still open

- **§0 needs hardware.** The RDNA 2 and gfx115x attention question is
  still inference from issue trackers. Run "Diagnose" on such a machine;
  it prints the answer directly.
- **`HSA_ENABLE_SDMA` unmeasured** (§5).
- **GGUF dequant kernels** (§3.3) need an upstream HIP build — not
  something the wrapper can fix.

---

## Sources

- [AMD Software: PyTorch on Windows Edition 7.2 Release Notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html) — Jan 21 2026; official Windows GPU compatibility list
- [Windows support matrices by ROCm version](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html) — FP8 = RDNA 4 only; "the entire ROCm stack is not yet supported on Windows"
- [ROCm Blogs — Attention Backends for AMD GPUs in ComfyUI](https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html) — SDPA vs FA vs Sage benchmarks on RDNA 3
- [ROCm/ROCm#5404 — AOTriton not available for gfx1151](https://github.com/ROCm/ROCm/issues/5404)
- [ROCm/TheRock#1364 — PyTorch Flash Attention with gfx1151](https://github.com/ROCm/TheRock/issues/1364) — `HSA_OVERRIDE_GFX_VERSION=11.0.0` workaround
- [ROCm/aotriton#16 — Memory Efficient Flash Attention for gfx1100](https://github.com/ROCm/aotriton/issues/16)
- [ROCm/aotriton architecture overview](https://deepwiki.com/ROCm/aotriton) — build target list
- [Wan2GP — docs/AMD-INSTALLATION.md](https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/AMD-INSTALLATION.md) — working Windows AMD Triton/Sage/FA2 recipe
- [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows) + [PyPI](https://pypi.org/project/triton-windows/) — `3.8.0.post28`, 29 Aug 2026; AMD support experimental
- [EmbeddedLLM/SageAttention-rocm](https://github.com/EmbeddedLLM/SageAttention-rocm) — ROCm Sage fork; no RDNA 3 benchmarks, last major update Nov 2024
- [Blizaine/Maestro](https://github.com/Blizaine/Maestro) — upstream `torch.js`, `app/shared/attention.py`, `app/scripts/install_*.py`
- [llama.cpp on Windows with AMD ROCm (HIP)](https://github.com/ggml-org/llama.cpp/discussions/27047) — working HIP backend, relevant to §3.3
