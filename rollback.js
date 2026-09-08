const fs = require("fs")
const path = require("path")
const { runtimeProfile } = require("./launcher_profile")

// Roll the upstream Maestro clone back to the revision it was on before
// the last Update. update.js records that SHA (via update_state.py
// record) immediately before `git reset --hard origin/HEAD`.
//
// Why this exists: upstream Maestro moves fast, and update.js always
// tracks its tip. When a bad upstream commit lands, every user's next
// Update breaks them — and the only other recovery path is Reset, which
// deletes every downloaded model and generated output. This gives back
// the last known-good state without touching user data.
//
// The SHA is read here in Node (rather than shelled out) so the menu can
// show whether a rollback point exists at all.

const PREV_HEAD = ".maestro_state/prev_head"

const readPrevHead = () => {
  try {
    const raw = fs.readFileSync(path.join(__dirname, PREV_HEAD), "utf8").trim()
    return /^[0-9a-f]{7,40}$/i.test(raw) ? raw : null
  } catch (e) {
    return null
  }
}

module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  const prev = readPrevHead()

  if (!prev) {
    return {
      run: [
        {
          method: "notify",
          params: {
            html: "No rollback point saved. One is recorded automatically each time you run Update.",
          },
          next: null,
        },
      ],
    }
  }

  return {
    run: [
      {
        method: "log",
        params: { raw: `Rolling Maestro back to ${prev.slice(0, 12)}...` },
      },
      {
        method: "shell.run",
        params: {
          path: "Maestro",
          message: `git reset --hard ${prev}`,
        },
      },
      // Dependencies may have changed between the two revisions, so
      // re-resolve them against the rolled-back requirements.txt.
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
          message: "uv pip install -r requirements.txt --index-strategy unsafe-best-match",
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
      // The UI bundle almost certainly no longer matches the rolled-back
      // source, so rebuild unconditionally here.
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
          title: "Rollback complete",
          description: `Maestro is back on ${prev.slice(0, 12)}. Running Update again will return you to the latest upstream version.`,
        },
      },
    ],
  }
}
