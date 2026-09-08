const fs = require("fs")
const path = require("path")
const { runtimeProfile } = require("./launcher_profile")

// Default runtime environment for `python launch.py`.
//
// TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: without this, PyTorch's SDPA
//   silently falls back to the naive "math" attention kernel on this ROCm
//   build (flash/mem-efficient are implemented via AOTriton but gated as
//   experimental). The math kernel materializes the full O(n^2) attention
//   matrix, which OOMs on anything but tiny sequence lengths — confirmed:
//   92 GiB requested for a 21K-token attention call that flash/
//   mem-efficient handle in under 150 MB. See CLAUDE.md #5.
//
//   Note this only helps on targets AOTriton actually ships kernels for.
//   RDNA 2 and (inconsistently) the gfx115x APUs get nothing from it —
//   PERFORMANCE.md §0. "Diagnose" reports which backends are live.
//
// The MIOpen/HIP vars below match wan2gp-amd's start.js — same ROCm
// nightly lineage, ported over for parity: expandable segments to cut
// allocator fragmentation stalls, MIOpen fast kernel-selection instead of
// exhaustive autotuning, and SDMA disabled to sidestep a known ROCm
// copy-engine slowdown/hang on some driver builds.
//
// PYTORCH_HIP_ALLOC_CONF=expandable_segments:True is a NO-OP on the ROCm
// Windows build. Confirmed on a live RX 7900 XTX (torch
// 2.10.0a0+rocm7.10.0), which warns at the first allocation:
//   UserWarning: expandable_segments not supported on this platform
//   (c10/hip/HIPAllocatorConfig.h:40)
// Harmless, and it still does something on Linux ROCm, so it stays — but
// do not count it as a Windows memory-fragmentation fix.
//
// HSA_ENABLE_SDMA=0 is the one worth questioning: disabling the DMA
// engines pushes host<->device copies onto other paths, and mmgp's whole
// offload strategy is heavy host<->device shuttling. On a driver without
// the SDMA bug it may cost throughput rather than save it. It is a
// one-line A/B via user_env.json — see PERFORMANCE.md §5.
const DEFAULT_ENV = {
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1",
  PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True",
  HSA_ENABLE_SDMA: "0",
  MIOPEN_FIND_MODE: "FAST",
  MIOPEN_DISABLE_CACHE: "1",
}

// Optional user overrides. Create user_env.json next to this file with a
// flat string->string object; entries win over DEFAULT_ENV, and a null
// value unsets a default entirely. Git-ignored, so it survives updates
// and never conflicts.
//
//   { "HSA_ENABLE_SDMA": null, "MIOPEN_FIND_MODE": "NORMAL" }
//
// This is the supported way to run the experiments in PERFORMANCE.md §5
// without editing tracked files.
const USER_ENV_FILE = "user_env.json"

const loadUserEnv = () => {
  const file = path.join(__dirname, USER_ENV_FILE)
  let raw
  try {
    raw = fs.readFileSync(file, "utf8")
  } catch (e) {
    return { overrides: {}, note: null }
  }
  try {
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { overrides: {}, note: `${USER_ENV_FILE} must be a JSON object — ignoring it.` }
    }
    return { overrides: parsed, note: null }
  } catch (e) {
    // Never let a typo in an optional config file block startup.
    return { overrides: {}, note: `${USER_ENV_FILE} is not valid JSON (${e.message}) — ignoring it.` }
  }
}

const buildEnv = (port) => {
  const { overrides, note } = loadUserEnv()
  const env = { SERVER_PORT: port, ...DEFAULT_ENV }
  const applied = []
  for (const [key, value] of Object.entries(overrides)) {
    if (value === null) {
      delete env[key]
      applied.push(`${key}=(unset)`)
    } else {
      env[key] = String(value)
      applied.push(`${key}=${value}`)
    }
  }
  return { env, note, applied }
}

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const port = await kernel.port()
  const { env, note, applied } = buildEnv(port)

  return {
    daemon: true,
    run: [
      // Guard: if the ROCm runtime marker is missing, tell the user to
      // run Install or Update first instead of failing inside Python.
      {
        when: `{{!exists('${runtime.marker}')}}`,
        method: "input",
        params: {
          title: "AMD ROCm runtime not installed",
          description: "Run Install (or Update) to set up Maestro's AMD ROCm environment, then start Maestro again.",
        },
        next: null,
      },
      // Preflight: catch a torch that isn't the ROCm build BEFORE handing
      // off to launch.py.
      //
      // The marker guard above only proves torch.js ran at some point. It
      // does not prove torch is still the ROCm wheel -- a later
      // requirements pass can silently replace it with the stock PyPI
      // build (CPU-only on Windows). When that happens Maestro imports
      // fine, prints "CUDA unavailable", and then dies ~40 frames deep in
      // models/wan/modules/t5.py with "Torch not compiled with CUDA
      // enabled", which tells the user nothing about the actual cause.
      // Reported by a real user. See CLAUDE.md "Known runtime issues" #7.
      //
      // Metadata-only and conservative: no torch import (startup is not
      // the place to pay for one), and an unrecognised version string is
      // treated as fine, so this can never block a working install.
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/verify_rocm_torch.py" --preflight`,
        },
      },
      {
        when: `{{exists('Maestro/app/${runtime.env}/.torch_broken')}}`,
        method: "input",
        params: {
          title: "Wrong PyTorch build installed — Maestro can't start",
          description:
            "The Python environment has a CPU-only or NVIDIA build of PyTorch instead of the AMD ROCm build, so Maestro cannot use your GPU. This usually happens when a dependency update replaces the ROCm wheels.\n\n" +
            "Fix it from the Pinokio menu:\n" +
            "  1. Click Update — it now detects this and reinstalls the ROCm wheels automatically.\n" +
            "  2. If that doesn't help, click Repair (rebuilds the environment, keeps your models).\n" +
            "  3. Click Diagnose at any point for a full report to attach to a bug report.\n\n" +
            "Starting anyway would fail with \"Torch not compiled with CUDA enabled\".",
        },
        next: null,
      },
      // Seed AMD-appropriate defaults for settings whose stock values are
      // NVIDIA-specific — most importantly the MiniMax H3 text encoder,
      // whose "NVFP4 AWQ (Recommended)" default hangs the whole OS on AMD
      // (CLAUDE.md #6). Writes a finetunes/ override, which is git-ignored
      // upstream and therefore survives `git reset --hard`. Touches no
      // upstream file. Idempotent and fast (plain JSON I/O).
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/configure_amd_defaults.py"`,
        },
      },
      ...(note
        ? [{ method: "log", params: { raw: `[start] ${note}` } }]
        : []),
      ...(applied.length
        ? [{ method: "log", params: { raw: `[start] ${USER_ENV_FILE} overrides: ${applied.join(", ")}` } }]
        : []),
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          env,
          path: "Maestro/app",
          message: "python launch.py",
          on: [
            { event: "/(http:\\/\\/[0-9.:]+)/", done: true },
          ],
        },
      },
      {
        method: "local.set",
        params: { url: "{{input.event[1]}}" },
      },
    ],
  }
}
