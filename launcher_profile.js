"use strict"

// AMD-only hardware routing. NVIDIA support lives in a sibling project.
// Pinokio exposes both a normalized gpu (`amd`) and a GFX architecture
// target (e.g. `gfx1100`) before Python/PyTorch exists, so this works
// for fresh installs as well as upgrades.

const isAmd = (kernel = {}) => kernel.gpu === "amd"

// ── GFX target resolution ────────────────────────────────────────────
// torch.js needs a GFX target to pick the right Windows wheel index.
// Pinokio's kernel.gpu_target detection (pinokiod's
// kernel/gpu/amd_gfx_targets.json name table) misses some real device
// names — most notably APU iGPU display names that Windows reports with
// a trailing "Graphics": Strix Halo's "AMD Radeon(TM) 8060S Graphics"
// normalizes to "radeon 8060s graphics", which isn't in the table, so
// gpu_target comes through null. resolveGpuTarget() trusts Pinokio's
// gpu_target when present and otherwise resolves from the raw GPU model
// strings. See CLAUDE.md "GPU detection" for the full failure history.

// Any real gfx identifier passes through — even unsupported ones, so
// torch.js can report exactly which target is missing.
const GFX_ID = /^gfx[0-9a-f]+$/i

// Fallback name→target table applied to raw GPU model strings when
// Pinokio's gpu_target is missing. Ordered APU-first (the class this
// fallback exists for), then dGPU families. dGPU entries map to the
// family's index-representative target — all variants in a family share
// one wheel index in torch.js, so family-level resolution is exact
// enough. Keep in sync with torch.js's WIN_WHEEL_INDEXES.
const GPU_NAME_TARGETS = [
  // Strix Halo APUs: Radeon 8060S / 8050S / 8040S iGPU → gfx1151
  { match: /\b(8060s|8050s|8040s)\b/i, target: "gfx1151" },
  // Strix Point APUs: Radeon 890M / 880M iGPU → gfx1150
  { match: /\b(890m|880m)\b/i, target: "gfx1150" },
  // Krackan Point APUs: Radeon 860M / 840M iGPU → gfx1152
  { match: /\b(860m|840m)\b/i, target: "gfx1152" },
  // RDNA 4 (RX 9000): gfx1200/1201 share the gfx120X-all index
  { match: /\brx\s*9\d{2,3}\b/i, target: "gfx1200" },
  // RDNA 3 dGPU (RX 7000/8000): gfx1100-1102 share the gfx110X-dgpu index
  { match: /\brx\s*[78]\d{2,3}\b/i, target: "gfx1100" },
  // RDNA 2 (RX 6000): gfx1030-1034 share the gfx103X-dgpu index
  { match: /\brx\s*6\d{2,3}\b/i, target: "gfx1030" },
]

const resolveGpuTarget = (kernel = {}) => {
  const fromPinokio = String(kernel.gpu_target || "").toLowerCase()
  if (GFX_ID.test(fromPinokio)) {
    return fromPinokio
  }

  // Fallback: scan the raw model strings, primary GPU first, then every
  // listed controller sorted by VRAM descending (so a real dGPU wins on
  // hybrid APU+dGPU systems).
  const candidates = []
  if (kernel.gpu_model) candidates.push(String(kernel.gpu_model))
  const gpus = Array.isArray(kernel.gpus) ? [...kernel.gpus] : []
  gpus.sort((a, b) => (Number(b && b.vram) || 0) - (Number(a && a.vram) || 0))
  for (const g of gpus) {
    if (g && g.model) candidates.push(String(g.model))
  }

  for (const name of candidates) {
    for (const { match, target } of GPU_NAME_TARGETS) {
      if (match.test(name)) return target
    }
  }
  return null
}

// RDNA 2 — RX 6000 series (gfx1030, gfx1031, gfx1032, gfx1034)
const isAmdRdna2 = (kernel = {}) =>
  isAmd(kernel) && /^gfx103[0124]$/.test(resolveGpuTarget(kernel) || "")

// RDNA 3 dGPU — RX 7000 / RX 8000 series (gfx1100, gfx1101, gfx1102).
// Excludes gfx1103 (Phoenix APU) intentionally — that target has its own
// nightly channel and different perf characteristics.
const isAmdRdna3 = (kernel = {}) =>
  isAmd(kernel) && /^gfx110[012]$/.test(resolveGpuTarget(kernel) || "")

// RDNA 4 — RX 9000 series (gfx1200, gfx1201)
const isAmdRdna4 = (kernel = {}) =>
  isAmd(kernel) && /^gfx120[01]$/.test(resolveGpuTarget(kernel) || "")

// AMD APUs — Strix Point (gfx1150), Strix Halo (gfx1151),
// Krackan Point (gfx1152), Krackan Halo (gfx1153).
const isAmdApu = (kernel = {}) =>
  isAmd(kernel) && /^gfx115[0-3]$/.test(resolveGpuTarget(kernel) || "")

// AMD ROCm runtime — single profile covers every supported AMD GPU.
// Windows uses per-gfx-target nightly wheels; Linux uses the stable
// rocm7.2 index. Python 3.11 required (ROCm wheels do not ship cp310
// builds).
//
// Venv layout pitfall: Pinokio resolves the `venv` param RELATIVE TO
// the step's `path` (pinokiod shell.js: env_path =
// path.resolve(params.path, params.venv)). Every step that touches the
// venv — including helper scripts that live in the wrapper root — MUST
// use path: runtime.path, and invoke its script by absolute path (see
// install.js/update.js). Using path: "." instead creates a stray venv
// at the app root, and the real venv never receives sitecustomize.py
// or ffmpeg — the exact failure behind the FSDP import crash on Start.
const amdRuntimeProfile = () => ({
  env: "env-amd",
  python: "3.11",
  path: "Maestro/app",
  marker: "Maestro/app/env-amd/.maestro_amd_v1.installed",
  label: "AMD ROCm",
})

// Kept as the single call site for other scripts even though there is
// only one profile — mirrors upstream Maestro's shape so future
// per-family branching (e.g. a distinct RDNA 4 profile) plugs in here
// without touching install/torch/start.
const runtimeProfile = () => amdRuntimeProfile()

module.exports = {
  isAmd,
  isAmdRdna2,
  isAmdRdna3,
  isAmdRdna4,
  isAmdApu,
  amdRuntimeProfile,
  runtimeProfile,
  resolveGpuTarget,
}
