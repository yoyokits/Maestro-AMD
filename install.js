const { runtimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
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
      // Wipe any leftover partial clone from an interrupted previous
      // install so the next step's `git clone` cannot fail on a
      // non-empty destination.
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
      // crashes the whole app at import time for a code path that's
      // never called. Drops a sitecustomize.py into the venv's
      // site-packages so CPython auto-loads it at every interpreter
      // startup and shadows the module before Maestro's own imports run.
      // Lives in the venv, so it survives Update (unlike a source patch,
      // which git reset --hard would wipe). See CLAUDE.md "Known runtime
      // issues" #1.
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          // MUST be the runtime venv's path, not ".": Pinokio resolves
          // `venv` relative to `path` (see launcher_profile.js). The
          // helper itself is invoked by absolute path (__dirname is the
          // wrapper root — install.js is required from there).
          path: runtime.path,
          message: `python "${__dirname}/install_sitecustomize.py"`,
        },
      },
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: "Maestro/app",
          message: [
            "uv pip install -r requirements.txt --index-strategy unsafe-best-match",
            "uv pip install hf-xet pip",
          ],
        },
      },
      // Ensure ffmpeg/ffprobe are on PATH. imageio-ffmpeg (a real
      // dependency) bundles its own ffmpeg but not under the plain name,
      // and ships no ffprobe at all — several other dependencies shell
      // out to literal `ffmpeg`/`ffprobe` commands. See CLAUDE.md "Known
      // runtime issues" #3/#4.
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/ensure_ffmpeg.py"`,
        },
      },
      {
        method: "script.start",
        params: {
          uri: "torch.js",
          params: {
            venv: runtime.env,
            venv_python: runtime.python,
            path: "Maestro/app",
          },
        },
      },
      // Fetch the seed-vc voice-conversion component (GPL-3.0). Lives
      // in its own repo — pinned to a tag for reproducible installs.
      // Bump the tag here AND in update.js together.
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
      {
        method: "input",
        params: {
          title: "Installation completed",
          description: 'Click "Start" to get started',
        },
      },
    ],
  }
}
