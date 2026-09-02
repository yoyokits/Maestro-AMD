module.exports = {
  version: "8.0",
  title: "Maestro AMD",
  description: "[AMD ONLY] All-in-one local AI video, image & music studio built on the WanGP pipeline (Wan 2.1/2.2, LTX-2.3, Qwen, Hunyuan Video, Flux). ROCm-powered. Supported on Windows and Linux for RDNA 2-4 dGPUs (RX 6000/7000/8000/9000) and gfx1150-53 APUs (Strix Point/Halo, Krackan Point/Halo). 6 GB+ VRAM recommended. For NVIDIA GPUs use the upstream Maestro app. Credits: the app itself is Maestro by Blizaine (github.com/Blizaine/Maestro) — this is just an AMD installer wrapper around it. The clone-at-install-time approach used to port it to AMD was pioneered by 6Morpheus6's wan2gp-amd (github.com/6Morpheus6/wan2gp-amd).",
  icon: "maestro_simplified_icon_alpha.png",
  menu: async (kernel, info) => {
    const installed = info.exists("Maestro/app/env-amd")
    const running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      start_latest: info.running("start_latest.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
    }

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    }

    if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Updating",
        href: "update.js",
      }]
    }

    if (running.reset) {
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Resetting",
        href: "reset.js",
      }]
    }

    if (installed) {
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

      return [{
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
      }, {
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
