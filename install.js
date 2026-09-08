const { runtimeProfile, resolveGpuTarget } = require("./launcher_profile")

// Windows wheel-index coverage, mirrored from torch.js's WIN_WHEEL_INDEXES
// so an unsupported card is rejected BEFORE the clone rather than after
// it. Keep the two in sync (see CLAUDE.md "GPU detection").
const WIN_SUPPORTED = /^(gfx103[0124]|gfx110[012]|gfx120[01]|gfx115[0-3])$/
const SUPPORTED_WIN_TARGETS =
  "RDNA 2 (gfx1030/31/32/34), RDNA 3 dGPU (gfx1100/01/02), RDNA 4 (gfx1200/01), APUs gfx1150/51/52/53"

// uv ships its own bundled CA roots (webpki) rather than using the OS
// trust store, so behind a corporate/AV TLS-inspection proxy it fails
// with `invalid peer certificate: UnknownIssuer` while git and curl on
// the same machine succeed — requirements.txt pulls several wheels by
// direct GitHub URL. UV_SYSTEM_CERTS switches uv to the OS store
// (schannel on Windows); it was UV_NATIVE_TLS before uv 0.11.
// UV_HTTP_TIMEOUT: uv defaults to 30s, and those GitHub release downloads
// measured 36s for a 61 KB file on a real connection. Both are harmless
// where they aren't needed. See CLAUDE.md #8.
const UV_ENV = { UV_SYSTEM_CERTS: "1", UV_HTTP_TIMEOUT: "180" }

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const gpuTarget = resolveGpuTarget(kernel)
  const isWin = kernel.platform === "win32"

  // Every venv-touching step uses this shape. `path` MUST be
  // runtime.path — Pinokio resolves `venv` relative to it, so path: "."
  // creates a stray venv at the app root and the real runtime venv never
  // receives the preamble or ffmpeg. The helper scripts live in the
  // wrapper root, so they are invoked by absolute path, with any
  // arguments kept OUTSIDE the quotes.
  // See CLAUDE.md "Venv layout pitfall".
  const helper = (script, args = "") => ({
    method: "shell.run",
    params: {
      venv: runtime.env,
      venv_python: runtime.python,
      path: runtime.path,
      message: `python "${__dirname}/${script}"${args ? " " + args : ""}`,
    },
  })

  const installTorch = {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: runtime.env,
        venv_python: runtime.python,
        path: runtime.path,
      },
    },
  }

  return {
    run: [
      {
        when: "{{gpu !== 'amd'}}",
        method: "notify",
        params: {
          html: "Maestro AMD requires an AMD GPU (RDNA 2 or newer, or a Strix Point/Halo/Krackan APU). For NVIDIA, use the upstream Maestro app.",
        },
        next: null,
      },
      {
        when: "{{platform !== 'win32' && platform !== 'linux'}}",
        method: "notify",
        params: {
          html: "AMD ROCm is only supported on Windows and Linux. macOS is not supported.",
        },
        next: null,
      },
      // Fail fast on an unsupported GPU. torch.js has the same check, but
      // it only runs after the clone, the venv and the full requirements
      // install — so without this an unsupported card burned ten-plus
      // minutes and several GB before dead-ending, leaving a venv with no
      // torch in it. Resolved in JS via resolveGpuTarget for the same
      // reason torch.js does it: Pinokio's gpu_target is null for some APU
      // names (CLAUDE.md "GPU detection").
      //
      // Windows only — torch.js's Linux branch uses the stable rocm7.2
      // index, which covers every ROCm-supported AMD GPU.
      ...(isWin && kernel.gpu === "amd" && !WIN_SUPPORTED.test(gpuTarget || "")
        ? [{
            method: "notify",
            params: {
              html: `Your AMD GPU${gpuTarget ? ` (target: ${gpuTarget})` : ""} is not supported on Windows yet. Supported: ${SUPPORTED_WIN_TARGETS}. Nothing has been installed.`,
            },
            next: null,
          }]
        : []),
      // Wipe any leftover partial clone from an interrupted previous
      // install so the next step's `git clone` cannot fail on a non-empty
      // destination.
      {
        method: "fs.rm",
        params: { path: "Maestro" },
      },
      {
        method: "shell.run",
        params: {
          message: "git clone https://github.com/Blizaine/Maestro Maestro",
        },
      },
      // Preemptively shadow torch.distributed.fsdp for a known upstream
      // bug: Maestro's models/wan/distributed/fsdp.py imports FSDP
      // unconditionally, and on PyTorch builds without a working
      // torch.distributed backend (e.g. this ROCm Windows wheel) that
      // crashes the whole app at import time for a code path that's never
      // called. Installs a .pth hook into the venv's site-packages so
      // CPython runs it at every interpreter startup, before Maestro's own
      // imports. Lives in the venv, so it survives Update (unlike a source
      // patch, which git reset --hard would wipe). See CLAUDE.md #1.
      helper("install_preamble.py"),
      // ROCm wheels BEFORE requirements.txt. Nothing in requirements.txt
      // pins torch, but torchcodec / accelerate / peft / timm /
      // open_clip_torch all depend on it — so resolving them against an
      // empty venv pulls the stock PyPI torch (CPU-only on Windows) just
      // to have torch.js --force-reinstall over the top. Installing ROCm
      // torch first leaves the requirement already satisfied.
      installTorch,
      {
        method: "shell.run",
        params: {
          env: UV_ENV,
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: [
            "uv pip install -r requirements.txt --index-strategy unsafe-best-match",
            "uv pip install hf-xet pip",
          ],
        },
      },
      // Safety net for the ordering above: if the resolver replaced the
      // ROCm build after all, this drops a flag file and the next step
      // reinstalls. Cheap — reads package metadata, and only imports torch
      // when the version string is ambiguous. See CLAUDE.md #7.
      helper("verify_rocm_torch.py"),
      {
        when: `{{exists('${runtime.path}/${runtime.env}/.torch_needs_reinstall')}}`,
        ...installTorch,
      },
      // Ensure ffmpeg/ffprobe are on PATH. imageio-ffmpeg (a real
      // dependency) bundles its own ffmpeg but not under the plain name,
      // and ships no ffprobe at all — several other dependencies shell out
      // to literal `ffmpeg`/`ffprobe` commands. See CLAUDE.md #3/#4.
      helper("ensure_ffmpeg.py"),
      // Replace NVIDIA-specific defaults that are wrong on AMD — most
      // importantly the MiniMax H3 text encoder, whose stock NVFP4 AWQ
      // value has hung whole machines. Writes a finetunes/ override, which
      // is git-ignored upstream and survives `git reset --hard`. See #6.
      helper("configure_amd_defaults.py"),
      // Fetch the seed-vc voice-conversion component (GPL-3.0). Lives in
      // its own repo — pinned to a tag for reproducible installs. Bump the
      // tag here AND in update.js together.
      {
        when: "{{!exists('Maestro/app/postprocessing/seedvc/__init__.py')}}",
        method: "shell.run",
        params: {
          message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc Maestro/app/postprocessing/seedvc",
        },
      },
      // Build the React Web UI. Without this, only the Classic UI
      // (Gradio, at /classic) is reachable.
      {
        when: "{{exists('Maestro/ui/package.json')}}",
        method: "shell.run",
        params: {
          path: "Maestro/ui",
          message: [
            "npm install",
            "npm run build",
          ],
        },
      },
      // Record which upstream revision the UI bundle was built from, so
      // Update can skip the npm rebuild when ui/ hasn't changed.
      helper("update_state.py", "ui-done"),
      {
        method: "input",
        params: {
          title: "Installation completed",
          description: 'Click "Start" to get started. If anything looks wrong, run "Diagnose" from the menu.',
        },
      },
    ],
  }
}
