const { runtimeProfile } = require("./launcher_profile")

const runtime = runtimeProfile()

module.exports = {
  version: "8.0",
  title: "Maestro AMD",
  description: "[AMD ONLY] All-in-one local AI video, image & music studio built on the WanGP pipeline (Wan 2.1/2.2, LTX-2.3, Qwen, Hunyuan Video, Flux). ROCm-powered. Supported on Windows and Linux for RDNA 2-4 dGPUs (RX 6000/7000/8000/9000) and gfx1150-53 APUs (Strix Point/Halo, Krackan Point/Halo). 6 GB+ VRAM recommended. For NVIDIA GPUs use the upstream Maestro app. Credits: the app itself is Maestro by Blizaine (github.com/Blizaine/Maestro) — this is just an AMD installer wrapper around it. The clone-at-install-time approach used to port it to AMD was pioneered by 6Morpheus6's wan2gp-amd (github.com/6Morpheus6/wan2gp-amd).",
  icon: "maestro_simplified_icon_alpha.png",
  menu: async (kernel, info) => {
    // The ROCm runtime marker, not the venv directory. A venv can exist
    // while the install is broken (an interrupted run, or torch.js
    // dead-ending on an unsupported GPU) — testing the directory offered
    // "Start", which start.js then immediately refused. The marker is
    // what start.js actually gates on, so the menu now agrees with it.
    const ready = info.exists(runtime.marker)
    const cloned = info.exists("Maestro/app")
    const canRollBack = info.exists(".maestro_state/prev_head")
    const inpaint = info.exists("Maestro/app/services/sam/env/.maestro-sam-ready")

    const running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      start_latest: info.running("start_latest.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      repair: info.running("repair.js"),
      rollback: info.running("rollback.js"),
      diagnose: info.running("diagnose.js"),
      sam: info.running("sam_install.js"),
    }

    const busy = [
      ["install", "Installing", "fa-solid fa-plug", "install.js"],
      ["update", "Updating", "fa-solid fa-terminal", "update.js"],
      ["reset", "Resetting", "fa-solid fa-terminal", "reset.js"],
      ["repair", "Repairing", "fa-solid fa-wrench", "repair.js"],
      ["rollback", "Rolling back", "fa-solid fa-clock-rotate-left", "rollback.js"],
      ["diagnose", "Diagnosing", "fa-solid fa-stethoscope", "diagnose.js"],
      ["sam", "Installing Inpaint", "fa-solid fa-wand-magic-sparkles", "sam_install.js"],
    ].find(([key]) => running[key])

    if (busy) {
      const [, text, icon, href] = busy
      return [{ default: true, icon, text, href }]
    }

    // Diagnose is useful in every state where there's something to look
    // at, including a half-finished install — that's exactly when people
    // need it most.
    const diagnose = {
      icon: "fa-solid fa-stethoscope",
      text: "Diagnose",
      href: "diagnose.js",
    }

    if (ready) {
      // start_latest.js triggers update.js and then start.js; while
      // update is still running, `running.start` is false but
      // start_latest owns the daemon slot. Once start.js takes over,
      // its local URL becomes available.
      if (running.start || running.start_latest) {
        const local = info.local("start.js")
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: "fa-solid fa-rocket",
            text: "Open Classic UI",
            href: local.url + "/classic",
          }, {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: running.start_latest ? "start_latest.js" : "start.js",
          }]
        }
        return [{
          default: true,
          icon: "fa-solid fa-terminal",
          text: running.start_latest ? "Updating & Starting" : "Starting",
          href: running.start_latest ? "start_latest.js" : "start.js",
        }]
      }

      const menu = [{
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      }, {
        icon: "fa-solid fa-rotate",
        text: "Update & Start",
        href: "start_latest.js",
      }, {
        icon: "fa-solid fa-plug",
        text: "Update",
        href: "update.js",
      }]

      if (canRollBack) {
        menu.push({
          icon: "fa-solid fa-clock-rotate-left",
          text: "<div><strong>Roll back last update</strong><div>Return to the previous Maestro version. Keeps models.</div></div>",
          href: "rollback.js",
          confirm: "Roll Maestro back to the version it was on before the last Update? Your models and outputs are kept.",
        })
      }

      menu.push(diagnose, {
        icon: "fa-solid fa-wand-magic-sparkles",
        text: inpaint ? "Reinstall Inpaint Support" : "Install Inpaint Support",
        href: "sam_install.js",
      }, {
        icon: "fa-solid fa-wrench",
        text: "<div><strong>Repair</strong><div>Rebuild the environment, keep models</div></div>",
        href: "repair.js",
        confirm: "Rebuild Maestro's environment? Downloaded models, LoRAs and outputs are preserved; everything else is reinstalled.",
      }, {
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
        href: "reset.js",
        confirm: "Are you sure you wish to reset the app? This deletes downloaded models and generated outputs. To fix a broken install without losing them, use Repair instead.",
      })

      return menu
    }

    // Cloned but never finished: offer the cheap fix (Repair keeps any
    // models already downloaded) before the expensive one.
    if (cloned) {
      return [{
        default: true,
        icon: "fa-solid fa-wrench",
        text: "<div><strong>Repair</strong><div>Finish the install, keep models</div></div>",
        href: "repair.js",
      }, diagnose, {
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
        href: "reset.js",
        confirm: "Are you sure you wish to reset the app? This deletes downloaded models and generated outputs.",
      }]
    }

    return [{
      default: true,
      icon: "fa-solid fa-plug",
      text: "Install",
      href: "install.js",
    }]
  },
}
