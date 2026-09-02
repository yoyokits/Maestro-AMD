const { runtimeProfile } = require("./launcher_profile")

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  return {
    run: [
      // Pull latest upstream Maestro. `git reset --hard origin/HEAD`
      // matches upstream's tracked files exactly while leaving untracked
      // user data (models, LoRAs, outputs, wgp_config.json — all
      // git-ignored inside Maestro/app/) intact.
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
      // Re-copy sitecustomize.py in case Reset (or the user) blew away
      // the venv -- see install.js and CLAUDE.md "Known runtime issues"
      // #1. Cheap; overwrites in place.
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          // MUST be the runtime venv's path, not ".": Pinokio resolves
          // `venv` relative to `path` (see launcher_profile.js). The
          // helper itself is invoked by absolute path (__dirname is the
          // wrapper root — update.js is required from there).
          path: runtime.path,
          message: `python "${__dirname}/install_sitecustomize.py"`,
        },
      },
      // Python deps may have moved between upstream commits.
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
      // Cheap no-op once already provisioned — see CLAUDE.md "Known
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
      // Skip the multi-GB ROCm wheel reinstall unless the marker is
      // gone (matches upstream Maestro's own update optimization).
      {
        when: `{{!exists('${runtime.marker}')}}`,
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
      // Self-heal the seedvc component if the user deleted it or an
      // earlier install was interrupted before this step ran.
      {
        when: "{{!exists('Maestro/app/postprocessing/seedvc/__init__.py')}}",
        method: "shell.run",
        params: {
          message: "git clone --depth 1 --branch v1.0.0 https://github.com/Blizaine/maestro-seedvc Maestro/app/postprocessing/seedvc",
        },
      },
      // Rebuild the React Web UI so it tracks upstream changes.
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
    ],
  }
}
