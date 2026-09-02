const { runtimeProfile, resolveGpuTarget } = require("./launcher_profile")

// AMD ROCm PyTorch install. The GFX target is resolved in JS from the
// Pinokio kernel object (see resolveGpuTarget in launcher_profile.js)
// instead of trusting the `gpu_target` template variable: Pinokio's
// built-in GPU-name detection misses some device names (Strix Halo APUs
// on Windows — its iGPU shows up as "AMD Radeon(TM) 8060S Graphics",
// and the trailing "Graphics" defeats pinokiod's name table), which
// previously made every wheel-install step silently skip and left the
// runtime marker unwritten.
//
// Windows: per-gfx-target nightly wheels from AMD's v2-staging index.
//   UV_SKIP_WHEEL_FILENAME_CHECK bypasses uv's filename validation
//   (nightly wheels don't match uv's expected pattern).
// Linux: stable rocm7.2 wheels from download.pytorch.org (torch 2.11 as
//   of Aug 2026 — see CLAUDE.md for the bump procedure).
//
// NVIDIA-only extras (SageAttention, FlashAttention, triton-windows,
// nunchaku, lightx2v, xformers) are intentionally skipped. WanGP's
// built-in SDPA path handles attention on AMD.

// Windows wheel index per gfx-target family. Variants share one index:
//   gfx1030-1034 → gfx103X-dgpu   (RDNA 2)
//   gfx1100-1102 → gfx110X-dgpu   (RDNA 3 dGPU; -dgpu faster than -all,
//                                  see ROCm/TheRock#3083)
//   gfx1200-1201 → gfx120X-all    (RDNA 4; -all is the only variant)
//   gfx1150/51/52/53 → per-target APU indexes
const WIN_WHEEL_INDEXES = {
  gfx1030: "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu",
  gfx1031: "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu",
  gfx1032: "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu",
  gfx1034: "https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu",
  gfx1100: "https://rocm.nightlies.amd.com/v2-staging/gfx110X-dgpu",
  gfx1101: "https://rocm.nightlies.amd.com/v2-staging/gfx110X-dgpu",
  gfx1102: "https://rocm.nightlies.amd.com/v2-staging/gfx110X-dgpu",
  gfx1200: "https://rocm.nightlies.amd.com/v2-staging/gfx120X-all",
  gfx1201: "https://rocm.nightlies.amd.com/v2-staging/gfx120X-all",
  gfx1150: "https://rocm.nightlies.amd.com/v2-staging/gfx1150",
  gfx1151: "https://rocm.nightlies.amd.com/v2-staging/gfx1151",
  gfx1152: "https://rocm.nightlies.amd.com/v2-staging/gfx1152",
  gfx1153: "https://rocm.nightlies.amd.com/v2-staging/gfx1153",
}

const SUPPORTED_WIN_TARGETS =
  "RDNA 2 (gfx1030/31/32/34), RDNA 3 dGPU (gfx1100/01/02), RDNA 4 (gfx1200/01), APUs gfx1150/51/52/53"

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const gpuTarget = resolveGpuTarget(kernel)
  const isWin = kernel.platform === "win32"

  // venv/venv_python/path come from install.js/update.js's script.start
  // params. `path` MUST resolve to the runtime venv's directory —
  // Pinokio resolves `venv` relative to it (see launcher_profile.js).
  const winWheel = (indexUrl) => ({
    method: "shell.run",
    params: {
      env: { UV_SKIP_WHEEL_FILENAME_CHECK: "1" },
      venv: "{{args && args.venv ? args.venv : null}}",
      venv_python: "{{args && args.venv_python ? args.venv_python : null}}",
      path: `{{args && args.path ? args.path : '${runtime.path}'}}`,
      message: `uv pip install --pre torch torchvision torchaudio --index-url ${indexUrl} --force-reinstall`,
    },
  })

  const winSteps = (() => {
    if (!isWin) return []
    const indexUrl = gpuTarget && WIN_WHEEL_INDEXES[gpuTarget]
    if (indexUrl) {
      return [
        {
          method: "log",
          params: { raw: `Resolved AMD GPU target: ${gpuTarget}` },
        },
        winWheel(indexUrl),
      ]
    }
    // Recognized-as-AMD but unknown/unsupported gfx target: stop with a
    // clear message instead of silently installing nothing.
    return [{
      method: "notify",
      params: {
        html: `Could not find a ROCm wheel index for your AMD GPU${gpuTarget ? ` (target: ${gpuTarget})` : ""}. Supported on Windows: ${SUPPORTED_WIN_TARGETS}. On Linux, any ROCm-supported AMD GPU works via the stable rocm7.2 wheel index.`,
      },
      next: null,
    }]
  })()

  const linuxSteps = (!isWin && kernel.platform === "linux") ? [{
    method: "shell.run",
    params: {
      venv: "{{args && args.venv ? args.venv : null}}",
      venv_python: "{{args && args.venv_python ? args.venv_python : null}}",
      path: `{{args && args.path ? args.path : '${runtime.path}'}}`,
      message: "uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2 --force-reinstall",
    },
  }] : []

  return {
    run: [
      {
        method: "log",
        params: { raw: `Installing Maestro's ${runtime.label} acceleration runtime...` },
      },

      // Platform-gated wheel install steps (built from the JS-resolved
      // gfx target above). Empty on unsupported platforms — the guard in
      // install.js already blocks non-Windows/Linux installs.
      ...winSteps,
      ...linuxSteps,

      // Runtime marker — update.js checks this to skip re-downloading
      // multi-GB ROCm wheels on routine updates. Deleting this file
      // forces a full reinstall on the next Update.
      {
        method: "fs.write",
        params: {
          path: runtime.marker,
          text: `Maestro ${runtime.label} runtime installed. Delete this file and run Update to reinstall it.`,
        },
      },
    ],
  }
}
