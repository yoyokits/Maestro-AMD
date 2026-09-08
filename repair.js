const { runtimeProfile } = require("./launcher_profile")

// "Repair (keep models)" — rebuild the environment without re-downloading
// user data.
//
// Reset deletes the whole Maestro/ tree, which means tens of GB of model
// checkpoints and every generated output. That is the right hammer for
// "start completely over", but it is far too blunt for the common case:
// a broken venv, a half-finished install, a corrupted node_modules.
//
// This does everything Install does except the clone and the wipe:
// `git clean -fdx` removes every untracked file — venv, node_modules,
// build output, stale caches — *except* the user-data paths listed below,
// then the normal install steps rebuild on top.
//
// Keep PRESERVE in sync with the git-ignored user-data paths documented
// in CLAUDE.md ("Layout"). Anything not listed here is treated as
// re-derivable and will be deleted.
const PRESERVE = [
  "app/models",
  "app/loras",
  "app/outputs",
  "app/wgp_config.json",
]

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const excludes = PRESERVE.map((p) => `-e ${p}`).join(" ")

  return {
    run: [
      {
        when: "{{!exists('Maestro/.git')}}",
        method: "notify",
        params: {
          html: "Maestro isn't installed yet — run Install first.",
        },
        next: null,
      },
      {
        method: "log",
        params: {
          raw: `Repairing install. Preserving: ${PRESERVE.join(", ")}`,
        },
      },
      // Restore tracked files to upstream, then strip every untracked
      // file except user data. -x includes git-ignored files (which is
      // where the venv and node_modules live); the -e patterns are what
      // keeps the expensive downloads.
      {
        method: "shell.run",
        params: {
          path: "Maestro",
          message: [
            "git fetch origin",
            "git reset --hard origin/HEAD",
            `git clean -fdx ${excludes}`,
          ],
        },
      },
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/install_preamble.py"`,
        },
      },
      // ROCm wheels before requirements — same reasoning as install.js.
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
      {
        method: "shell.run",
        params: {
          // uv ships its own bundled CA roots (webpki) instead of using the
          // OS trust store, so on any machine behind a corporate/AV TLS-
          // inspection proxy it fails with `invalid peer certificate:
          // UnknownIssuer` while git and curl on the same machine succeed.
          // Observed live here against github.com release wheels
          // (smplfitter, chumpy) that requirements.txt pulls by direct URL.
          // Same root cause as CLAUDE.md #3/#4. UV_SYSTEM_CERTS makes uv
          // use the OS store (schannel on Windows); it was called
          // UV_NATIVE_TLS before uv 0.11 and is harmless on machines that
          // don't need it.
          //
          // UV_HTTP_TIMEOUT: uv defaults to 30s. Those same GitHub release
          // downloads measured 36s for a 61 KB file on a normal connection
          // here, which is enough to fail the whole install. See
          // CLAUDE.md #8.
          env: { UV_SYSTEM_CERTS: "1", UV_HTTP_TIMEOUT: "180" },
          venv: runtime.env,
          venv_python: runtime.python,
          path: "Maestro/app",
          message: [
            "uv pip install -r requirements.txt --index-strategy unsafe-best-match",
            "uv pip install hf-xet pip",
          ],
        },
      },
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/verify_rocm_torch.py"`,
        },
      },
      {
        when: `{{exists('Maestro/app/${runtime.env}/.torch_needs_reinstall')}}`,
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
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/ensure_ffmpeg.py"`,
        },
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
      {
        when: "{{!exists('Maestro/app/postprocessing/seedvc/__init__.py')}}",
        method: "shell.run",
        params: {
          message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc Maestro/app/postprocessing/seedvc",
        },
      },
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
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          path: runtime.path,
          message: `python "${__dirname}/update_state.py" ui-done`,
        },
      },
      {
        method: "input",
        params: {
          title: "Repair complete",
          description: "Your models and outputs were preserved. Click Start, or run Diagnose if problems persist.",
        },
      },
    ],
  }
}
