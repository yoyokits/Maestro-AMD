const { runtimeProfile } = require("./launcher_profile")

// See install.js for why these two uv settings exist (CLAUDE.md #8).
const UV_ENV = { UV_SYSTEM_CERTS: "1", UV_HTTP_TIMEOUT: "180" }

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)

  // Progress announcements. Pinokio writes each shell.run step to its own
  // log file and the terminal shows only the current step, so a long
  // silent step (uv resolve, npm install) looks like a freeze. These cost
  // nothing and tell the user which phase they are in.
  let phaseNo = 0
  const PHASES = 9
  const phase = (text) => ({
    method: "log",
    params: { raw: `=== [${++phaseNo}/${PHASES}] ${text} ===` },
  })

  // path MUST be runtime.path — Pinokio resolves `venv` relative to it.
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

  return {
    run: [
      // Nothing to update if Install never ran (or Reset wiped it).
      {
        when: "{{!exists('Maestro/.git')}}",
        method: "notify",
        params: { html: "Maestro isn't installed yet — run Install first." },
        next: null,
      },
      // Save the current upstream revision before it's overwritten so
      // "Roll back last update" has somewhere to go. Upstream Maestro
      // moves fast; without this a bad upstream commit leaves Reset —
      // which deletes every downloaded model — as the only escape.
      phase("Saving a rollback point"),
      helper("update_state.py", "record"),
      // Pull latest upstream Maestro. `git reset --hard origin/HEAD`
      // matches upstream's tracked files exactly while leaving untracked
      // user data (models, LoRAs, outputs, wgp_config.json — all
      // git-ignored inside Maestro/app/) intact.
      phase("Fetching the latest Maestro"),
      {
        method: "shell.run",
        params: {
          path: "Maestro",
          message: [
            "git fetch origin",
            "git reset --hard origin/HEAD",
          ],
        },
      },
      // Re-install the startup preamble in case Reset (or the user) blew
      // away the venv — see install.js and CLAUDE.md #1. Cheap; overwrites
      // in place. Also migrates installs still carrying the older
      // sitecustomize.py hook.
      phase("Installing the startup preamble"),
      helper("install_preamble.py"),
      phase("Installing Python dependencies (this is the slow one)"),
      // Python deps may have moved between upstream commits.
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
      // Catch the case where the requirements pass replaced the ROCm torch
      // with a CUDA or CPU build (a new upstream pin can do this). Writes
      // .torch_needs_reinstall, which the next step acts on. CLAUDE.md #7.
      phase("Verifying the ROCm PyTorch build"),
      helper("verify_rocm_torch.py"),
      // Skip the multi-GB ROCm wheel reinstall unless the marker is gone
      // or the verify step above found the wrong build.
      {
        when: `{{!exists('${runtime.marker}') || exists('${runtime.path}/${runtime.env}/.torch_needs_reinstall')}}`,
        method: "script.start",
        params: {
          uri: "torch.js",
          params: {
            venv: runtime.env,
            venv_python: runtime.python,
            path: runtime.path,
          },
        },
      },
      // Cheap no-op once already provisioned — see CLAUDE.md #3/#4.
      phase("Checking ffmpeg / ffprobe"),
      helper("ensure_ffmpeg.py"),
      // Re-assert AMD-appropriate defaults; upstream may have added new
      // model variants that need the same override. See CLAUDE.md #6.
      phase("Applying AMD-specific defaults"),
      helper("configure_amd_defaults.py"),
      // Self-heal the seedvc component if the user deleted it or an
      // earlier install was interrupted before this step ran.
      {
        when: "{{!exists('Maestro/app/postprocessing/seedvc/__init__.py')}}",
        method: "shell.run",
        params: {
          message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc Maestro/app/postprocessing/seedvc",
        },
      },
      // Decide whether the React UI actually needs rebuilding. `npm install
      // && npm run build` is the slowest step in an update and most
      // upstream commits don't touch ui/ at all — this makes the everyday
      // "Update & Start" path much faster. Fails open (schedules the
      // rebuild) on any doubt.
      phase("Checking whether the web UI needs rebuilding"),
      helper("update_state.py", "ui-check"),
      {
        when: "{{exists('Maestro/ui/package.json') && exists('.maestro_state/ui_rebuild')}}",
        method: "log",
        params: { raw: "=== [9/9] Rebuilding the web UI (npm - takes a minute) ===" },
      },
      {
        when: "{{exists('Maestro/ui/package.json') && exists('.maestro_state/ui_rebuild')}}",
        method: "shell.run",
        params: {
          path: "Maestro/ui",
          message: [
            "npm install",
            "npm run build",
          ],
        },
      },
      helper("update_state.py", "ui-done"),
      {
        method: "log",
        params: { raw: "=== Update complete. Click Start, or Diagnose if anything looks wrong. ===" },
      },
    ],
  }
}
