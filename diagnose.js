const { runtimeProfile } = require("./launcher_profile")

// Health check. Runs diagnose.py inside the ROCm venv with the same
// environment start.js uses, so the attention-backend probe reflects what
// Maestro actually runs with rather than a bare shell.
//
// Read-only: installs nothing, changes nothing, safe at any time.
module.exports = async (kernel) => {
  const runtime = runtimeProfile(kernel)
  return {
    run: [
      {
        when: "{{!exists('Maestro/app')}}",
        method: "notify",
        params: {
          html: "Maestro isn't installed yet — run Install first.",
        },
        next: null,
      },
      {
        method: "shell.run",
        params: {
          venv: runtime.env,
          venv_python: runtime.python,
          // Mirror start.js so the SDPA probe sees the real conditions.
          env: {
            TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL: "1",
            PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True",
            HSA_ENABLE_SDMA: "0",
            MIOPEN_FIND_MODE: "FAST",
            MIOPEN_DISABLE_CACHE: "1",
          },
          path: runtime.path,
          message: `python "${__dirname}/diagnose.py"`,
        },
      },
      {
        method: "input",
        params: {
          title: "Diagnostics complete",
          description: "Scroll up in this terminal for the report. Include it when reporting an issue.",
        },
      },
    ],
  }
}
