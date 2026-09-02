"use strict"
// Unit test for launcher_profile.resolveGpuTarget + torch.js step shape.
// Run: node test_gpu_detection.js
const assert = require("assert")
const { resolveGpuTarget, isAmd, isAmdApu } = require("./launcher_profile")
const torchConfig = require("./torch.js")

const kernelStrixHalo = {
  // Exact profile of the failing machine: Pinokio's gpu_target is null
  // because "AMD Radeon(TM) 8060S Graphics" isn't in pinokiod's table.
  platform: "win32",
  gpu: "amd",
  gpu_model: "amd radeon(tm) 8060s graphics",
  gpu_target: null,
  gpus: [{ name: "advanced micro devices, inc.", model: "amd radeon(tm) 8060s graphics", vram: 98304 }],
}

const kernelPinokioWorks = {
  platform: "win32",
  gpu: "amd",
  gpu_model: "amd radeon rx 7900 xtx",
  gpu_target: "gfx1100",
  gpus: [{ name: "advanced micro devices, inc.", model: "amd radeon rx 7900 xtx", vram: 24560 }],
}

// Pinokio resolved an exact target that has no Windows wheel index
// (Phoenix APU, gfx1103) — must notify and stop, not silently skip.
const kernelUnsupported = {
  platform: "win32",
  gpu: "amd",
  gpu_model: "amd radeon(tm) 780m",
  gpu_target: "gfx1103",
  gpus: [{ name: "advanced micro devices, inc.", model: "amd radeon(tm) 780m", vram: 8192 }],
}

const kernelLinux = {
  platform: "linux",
  gpu: "amd",
  gpu_model: "amd radeon rx 7900 xtx",
  gpu_target: "gfx1100",
  gpus: [],
}

const kernelUnknownName = {
  platform: "win32",
  gpu: "amd",
  gpu_model: "amd radeon graphics",
  gpu_target: null,
  gpus: [{ name: "advanced micro devices, inc.", model: "amd radeon graphics", vram: 8192 }],
}

// ── resolveGpuTarget ─────────────────────────────────────────────────
assert.strictEqual(resolveGpuTarget(kernelStrixHalo), "gfx1151", "Strix Halo fallback")
assert.strictEqual(resolveGpuTarget(kernelPinokioWorks), "gfx1100", "Pinokio passthrough")
assert.strictEqual(resolveGpuTarget({ gpu_model: "amd radeon rx 6300" }), "gfx1030", "RDNA2 family-level resolution")
assert.strictEqual(resolveGpuTarget({}), null, "empty kernel")
assert.strictEqual(resolveGpuTarget(kernelUnknownName), null, "unmatched name stays null")

// Variant names a fallback should handle (other APU SKUs, dGPU spellings)
assert.strictEqual(resolveGpuTarget({ gpu_model: "AMD Radeon(TM) 8050S Graphics" }), "gfx1151")
assert.strictEqual(resolveGpuTarget({ gpu_model: "AMD Radeon 890M" }), "gfx1150")
assert.strictEqual(resolveGpuTarget({ gpu_model: "AMD Radeon(TM) 860M" }), "gfx1152")
assert.strictEqual(resolveGpuTarget({ gpu_model: "AMD Radeon RX 9070 XT" }), "gfx1200")
assert.strictEqual(resolveGpuTarget({ gpu_model: "AMD Radeon RX 7600" }), "gfx1100")
assert.strictEqual(resolveGpuTarget({ gpu_target: "GFX1151" }), "gfx1151", "case-insensitive passthrough")
assert.strictEqual(resolveGpuTarget(kernelUnsupported), "gfx1103", "exact unsupported passthrough")
assert.strictEqual(resolveGpuTarget({
  gpu_model: "amd radeon(tm) 8060s graphics",
  gpus: [
    { model: "amd radeon(tm) 8060s graphics", vram: 98304 },
    { model: "amd radeon rx 7900 xtx", vram: 24560 },
  ],
}), "gfx1151", "primary gpu_model wins over gpus list")

// isAmdApu now works through the fallback
assert.strictEqual(isAmdApu(kernelStrixHalo), true, "isAmdApu via fallback")

// ── torch.js step shape ──────────────────────────────────────────────
;(async () => {
  let cfg = await torchConfig(kernelStrixHalo)
  let steps = cfg.run
  assert.strictEqual(steps[0].method, "log")
  const shellSteps = steps.filter(s => s.method === "shell.run")
  assert.strictEqual(shellSteps.length, 1, "exactly one wheel install step")
  assert.ok(shellSteps[0].params.message.includes("rocm.nightlies.amd.com/v2-staging/gfx1151"), "gfx1151 index")
  assert.strictEqual(shellSteps[0].params.env.UV_SKIP_WHEEL_FILENAME_CHECK, "1")
  // Venv-path contract: the wheel step's `path` template must fall back
  // to the runtime venv dir (Maestro/app), never ".". Pinokio resolves
  // `venv` relative to `path`; a "." fallback would recreate the stray
  // app-root venv bug. The literal must be baked in at build time since
  // template memory has no `runtime` binding.
  assert.ok(shellSteps[0].params.path.includes("'Maestro/app'"), "win wheel path falls back to runtime venv dir")
  assert.ok(!shellSteps[0].params.path.includes("runtime.path"), "no unresolved JS identifier left in template")
  assert.strictEqual(steps[steps.length - 1].method, "fs.write", "marker last")

  cfg = await torchConfig(kernelPinokioWorks)
  const shell7900 = cfg.run.filter(s => s.method === "shell.run")
  assert.strictEqual(shell7900.length, 1)
  assert.ok(shell7900[0].params.message.includes("gfx110X-dgpu"), "RDNA3 dgpu index")

  cfg = await torchConfig(kernelUnsupported)
  assert.ok(cfg.run.some(s => s.method === "notify" && s.next === null), "unsupported target notifies and stops")

  cfg = await torchConfig(kernelUnknownName)
  assert.ok(cfg.run.some(s => s.method === "notify" && s.next === null), "unresolved target notifies and stops")
  assert.ok(cfg.run.find(s => s.method === "notify").params.html.includes("Could not find"), "clear message")

  cfg = await torchConfig(kernelLinux)
  const linuxShell = cfg.run.filter(s => s.method === "shell.run")
  assert.strictEqual(linuxShell.length, 1)
  assert.ok(linuxShell[0].params.message.includes("download.pytorch.org/whl/rocm7.2"), "linux stable index")
  assert.ok(linuxShell[0].params.message.includes("torch==2.11.0"), "linux pinned versions")
  assert.ok(linuxShell[0].params.path.includes("'Maestro/app'"), "linux path falls back to runtime venv dir")

  console.log("ALL TESTS PASSED")
})().catch((e) => { console.error("TEST FAILED:", e.message); process.exit(1) })
