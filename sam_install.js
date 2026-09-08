const { resolveGpuTarget } = require("./launcher_profile")
const { WIN_WHEEL_INDEXES, LINUX_INDEX_URL, LINUX_TORCH_PINS } = require("./torch.js")

// SAM 3.1 segmentation service — AMD/ROCm port of upstream Maestro's
// sam_install.js. Powers the experimental Inpaint mode.
//
// Why this is a separate script and a separate environment: SAM 3.1 needs
// Python 3.12 and NumPy 1.x, both of which conflict with Maestro's own
// Python 3.11 / NumPy 2.x runtime. Upstream solves that with a dedicated
// conda env, and that isolation is exactly what makes this safe to add
// here — nothing it installs can reach or break the ROCm venv the rest of
// the wrapper depends on.
//
// The only AMD-specific change from upstream is the torch index URL:
// upstream hardcodes CUDA 12.8, this resolves the right ROCm index for
// the current GPU from launcher_profile.js. Everything else — the SAM3
// clone, the dependency resolution, the import health check, the ready
// marker — is vendor-neutral and carried over as-is.
//
// Not installed by default: it adds several minutes and a second torch
// download to a fresh install, and Inpaint is gated behind the
// experimental flag in Settings → Services anyway. Users opt in from the
// Pinokio menu.
//
// Status: experimental on AMD. SAM 3.1 itself is ordinary PyTorch and has
// no CUDA-only kernels, so it is expected to work — but it is not part of
// the verified path. Report failures rather than assuming they're yours.

const SAM = "Maestro/app/services/sam"

// Resolve the same ROCm wheel index torch.js uses, from the same table,
// so SAM's separate env can't drift from the runtime's. gfx target comes
// from resolveGpuTarget for the reason described in CLAUDE.md
// "GPU detection" — Pinokio's gpu_target is null for some APU names.
const rocmWheelIndex = (kernel = {}) => {
  if (kernel.gpu !== "amd") return null
  if (kernel.platform === "linux") {
    return { url: LINUX_INDEX_URL, pins: LINUX_TORCH_PINS, pre: false }
  }
  if (kernel.platform === "win32") {
    const url = WIN_WHEEL_INDEXES[resolveGpuTarget(kernel) || ""]
    return url ? { url, pins: "torch torchvision", pre: true } : null
  }
  return null
}

module.exports = async (kernel) => {
  const rocm = rocmWheelIndex(kernel)

  if (!rocm) {
    return {
      run: [
        {
          method: "notify",
          params: {
            html: "Inpaint support needs a supported AMD GPU on Windows or Linux. Run Diagnose to see what was detected.",
          },
          next: null,
        },
      ],
    }
  }

  const torchInstall = [
    "python -m pip install",
    rocm.pre ? "--pre" : "",
    // SAM only needs torch + torchvision; torchaudio is Maestro's, not its.
    rocm.pins.replace(/\s*torchaudio\S*/, ""),
    `--index-url ${rocm.url}`,
  ]
    .filter(Boolean)
    .join(" ")

  return {
    run: [
      {
        when: "{{!exists('Maestro/.git')}}",
        method: "notify",
        params: { html: "Maestro isn't installed yet — run Install first." },
        next: null,
      },
      // A previous successful marker must not survive a failed reinstall.
      // The menu treats SAM as installed only after the checks below pass.
      {
        when: `{{exists('${SAM}/env/.maestro-sam-ready')}}`,
        method: "fs.rm",
        params: { path: `${SAM}/env/.maestro-sam-ready` },
      },
      {
        when: `{{!exists('${SAM}/sam3')}}`,
        method: "shell.run",
        params: {
          message: `git clone https://github.com/facebookresearch/sam3.git ${SAM}/sam3`,
        },
      },
      {
        when: `{{exists('${SAM}/sam3')}}`,
        method: "shell.run",
        params: {
          path: `${SAM}/sam3`,
          message: "git pull",
        },
      },
      // ROCm torch in the isolated Python 3.12 conda env.
      {
        method: "shell.run",
        params: {
          env: { UV_SKIP_WHEEL_FILENAME_CHECK: "1" },
          conda: { path: `${SAM}/env`, python: "3.12" },
          message: torchInstall,
        },
      },
      // Resolve SAM3 and its runtime dependencies in a single transaction.
      // Upstream learned the hard way that splitting these lets a later
      // OpenCV/SciPy pass silently replace SAM3's required NumPy 1.x with
      // 2.x while pip still exits successfully — hence the explicit
      // version assertion at the end.
      {
        method: "shell.run",
        params: {
          conda: { path: `${SAM}/env`, python: "3.12" },
          message: [
            `python -m pip install -r ${SAM}/requirements.txt ${SAM}/sam3`,
            "python -m pip check",
            "python -c \"import numpy as np, cv2, scipy; from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor; from sam3.model.sam3_image_processor import Sam3Processor; major = int(np.__version__.split('.')[0]); assert major == 1, f'Expected NumPy 1.x, found {np.__version__}'; print(f'SAM dependency check passed: numpy={np.__version__}, opencv={cv2.__version__}, scipy={scipy.__version__}')\"",
          ],
        },
      },
      // Confirm the ROCm build actually survived dependency resolution —
      // same failure mode verify_rocm_torch.py guards in the main venv.
      {
        method: "shell.run",
        params: {
          conda: { path: `${SAM}/env`, python: "3.12" },
          message: "python -c \"import torch; print('SAM torch:', torch.__version__, 'hip:', torch.version.hip, 'gpu:', torch.cuda.is_available())\"",
        },
      },
      // Written only after every command above succeeds, so a broken or
      // incompatible env stays visibly repairable in Pinokio.
      {
        method: "fs.write",
        params: {
          path: `${SAM}/env/.maestro-sam-ready`,
          text: "SAM 3.1 dependency health check passed (AMD/ROCm).\n",
        },
      },
      {
        method: "input",
        params: {
          title: "Inpaint support installed",
          description: "Enable it in Maestro under Settings → Services. Model checkpoints (~1.7 GB) download on first use.",
        },
      },
    ],
  }
}
