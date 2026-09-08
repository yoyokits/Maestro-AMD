// Full revert to pre-install state: removes the upstream Maestro clone
// (which holds the real env-amd venv), all downloaded models / outputs
// / config stored inside Maestro/, and the stray app-root `env-amd`
// that pre-fix install/update runs created (Pinokio resolves `venv`
// relative to a step's `path`, and the helper steps used to run with
// path: "." — see launcher_profile.js). Menu already gates this behind
// a confirm.
module.exports = {
  run: [
    { method: "fs.rm", params: { path: "Maestro" } },
    { method: "fs.rm", params: { path: "env-amd" } },
    // Wrapper-local state (rollback point, UI build record). Lives
    // outside Maestro/ so repair.js's git clean cannot destroy it —
    // but a full Reset should clear it.
    { method: "fs.rm", params: { path: ".maestro_state" } },
  ],
}
