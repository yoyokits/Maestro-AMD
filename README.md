<p align="center">
    <img src="maestro_simplified_icon_alpha.png" alt="Maestro AMD icon" width="150">
</p>

# Maestro AMD

**A 100% local AI video, image & music studio — for AMD GPU owners.**

Maestro AMD is a [Pinokio](https://pinokio.computer) app that installs and runs
[Maestro](https://github.com/Blizaine/Maestro) — an all-in-one creative
studio for generating video, images, and audio entirely on your own PC — on
AMD graphics cards via ROCm. The upstream Maestro project only supports
NVIDIA; this wrapper adds AMD support without changing anything about how
Maestro itself works. No cloud, no subscription, no account required.

## What is Maestro?

Maestro turns text prompts (and images) into finished video, image, and
audio content using open-source AI models running on your own hardware.
Everything happens locally — your prompts, your media, and your generated
files never leave your machine.

### Director Mode — one prompt, a finished piece

Describe what you want and an LLM plans it out shot-by-shot for you:

- **Music Video** — feed it a song; it reads the BPM and structure, times
  cuts to the beat, and can target specific vocal segments.
- **Short Film** — give it a premise; it writes a screenplay with named,
  consistent characters and continuous dialogue, then shoots it scene by
  scene.

You don't need to know anything about prompting individual shots — Director
mode handles planning, then generation, automatically.

### Studio Mode — full manual control

For hands-on creators, Studio mode gives direct access to the underlying
generation models:

- **Video**: MiniMax H3, LTX-2.5 / LTX-2.3, Wan 2.1/2.2, Hunyuan Video
- **Image**: Flux 2 Klein, Krea 2, Qwen Image Edit
- **Audio**: text-to-speech (Kugelaudio, Qwen3), music generation
  (MiniMax-Music3, ACE-Step), sound effects (MMAudio)
- Multi-clip generation with smooth transitions, and character continuity
  across shots via keyframe injection

### Edit Mode — fix things after the fact

- **Retake** — re-roll just one part of a clip you didn't like
- **Edit Anything** — change an element with a text prompt
- **Outpaint** — extend a clip while keeping the action and audio flowing
- **Repaint** — restyle the visuals while keeping the original motion
- **Recast** — swap a character for someone else across an entire scene

### Everything else

- **Performance auto-tune** — detects your GPU and RAM on launch and picks
  sensible quality/quantization settings automatically. You shouldn't need
  to hand-tune anything to get started.
- **Built-in local LLM** — Director mode's planning brain downloads and runs
  itself (small, GGUF-based); no external API key needed, though OpenAI/
  Anthropic-compatible endpoints are supported if you'd rather use those.
- **CivitAI LoRA browser** — search, install, and manage LoRAs from inside
  the app, with AI-written usage guides for each one.
- **Workspaces** — keep separate projects and output folders isolated from
  each other.
- Light/dark themes, an opt-in NSFW mode, and a dashboard of everything
  you've generated so far.

## AMD GPU support

Upstream Maestro is CUDA-only. This wrapper installs
[PyTorch built for ROCm](https://rocm.docs.amd.com/) instead, so the exact
same models and features run on AMD hardware.

**Supported GPUs:**

| Family | Cards |
|---|---|
| RDNA 2 | RX 6600 – RX 6950 XT |
| RDNA 3 | RX 7600 – RX 7900 XTX, RX 8000 series |
| RDNA 4 | RX 9000 series |
| Ryzen AI APUs | Strix Point / Strix Halo / Krackan Point / Krackan Halo (Radeon 800/8060S-class integrated graphics) |

Windows and Linux are both supported. macOS is not (ROCm doesn't run there).

### Honest performance notes

- **Output quality is identical to the NVIDIA version.** Quality comes from
  the model weights, not the GPU brand — an AMD card and an NVIDIA card
  running the same model produce the same kind of results.
- **Speed is usually a bit behind an equivalent-tier NVIDIA card.** NVIDIA
  has optimized attention kernels (FlashAttention, SageAttention) that have
  no ROCm equivalent yet, so this wrapper falls back to PyTorch's standard
  attention path. It works, it's just not the fastest path available on
  NVIDIA.
- **Windows ROCm support is newer and less battle-tested than Linux.** AMD
  ships Windows PyTorch wheels from a nightly/staging channel (there's no
  stable Windows ROCm PyTorch release yet as of this writing), so expect
  occasional rough edges — a driver or nightly-wheel update can shift
  things. Linux uses AMD's stable ROCm wheel index and is generally the
  smoother experience if you have the choice.
- **VRAM matters more than raw speed for whether something works at all.**
  Rough guide, mirroring what you'd see on the NVIDIA side:

  | VRAM | What to expect |
  |---|---|
  | 24 GB+ (e.g. RX 7900 XTX) | Everything runs comfortably; the reference tier this app is tuned against |
  | 12–16 GB | Automatic offloading kicks in; noticeably slower but works for most models |
  | 6–8 GB | Usable for smaller/shorter generations with heavy offloading; expect long waits for anything ambitious |

- **Disk space**: budget 50–100 GB for an initial useful set of models; the
  full model collection across every feature exceeds 300 GB. Nothing
  downloads until you actually pick a model to use.

### ⚠️ Before using MiniMax H3: change one setting or risk a system hang

Maestro's **MiniMax H3** models (Omni, First/Last, Full, etc.) default to an
**NVFP4** text encoder — an NVIDIA-only format, despite being labeled
"Recommended" in the UI. On AMD there's no GPU kernel for it, so it silently
falls back to running a 32-billion-parameter model on your **CPU**. Combined
with how much RAM the rest of the model needs, this can push your system into
heavy memory-swap territory — bad enough that Windows itself becomes
unresponsive and won't recover without a hard reset. This isn't something the
AMD wrapper can fix for you automatically — it's a per-generation setting
inside Maestro itself.

**Fix — do this once before your first MiniMax H3 generation:**

1. Select any MiniMax H3 model, then open **Advanced Settings**.
2. Find **"H3 Text Encoder"** and change it from **NVFP4 AWQ (Recommended)**
   to **GGUF Q4_K_M** (or **GGUF Q2_K** if you're tight on RAM). The
   "Recommended" label doesn't account for AMD GPUs — GGUF is the one that
   actually runs correctly here.
3. In **Settings → Services**, leave **LLM Device** set to **CPU** — that's
   the correct choice for everyone (not AMD-specific), since it keeps the
   local Director planning LLM off your GPU so it doesn't compete with video
   generation for VRAM.

## Installing

1. Install [Pinokio](https://pinokio.computer) if you don't already have it.
2. Point Pinokio at this repository (or search Discover for "Maestro AMD"
   once it's listed).
3. Click **Install**. This clones Maestro, sets up a Python environment,
   installs the AMD ROCm build of PyTorch for your specific GPU, and builds
   the web UI. Takes roughly 10–20 minutes depending on your connection —
   this does *not* download model weights yet, those come later, per model,
   the first time you use one.
4. Click **Start**. A browser tab opens with the Maestro UI once the server
   is ready.

### Keeping it updated

- **Start** — launch with what's currently installed.
- **Update & Start** — pull the latest Maestro and launch in one click. This
  is the easiest way to stay current; recommended as your everyday launch
  button.
- **Update** — just pull the latest Maestro without starting it.
- **Reset** — removes Maestro and the Python environment entirely (a clean
  slate). Your downloaded models, LoRAs, and generated outputs live inside
  the same folder Reset deletes, so back them up first if you want to keep
  them.

## Licensing

Maestro is released under the **WanGP Non-Commercial Evaluation License
1.1**. Generated outputs are yours to use commercially with attribution;
using the *software itself* commercially requires separate licensing from
its author. Individual models it uses (MiniMax H3, Flux, Qwen, Gemma, etc.)
keep their own original licenses. The seed-vc voice-conversion component is
GPL-3.0 and lives in its own repository.

This wrapper (the Pinokio scripts in this repository) doesn't modify or
redistribute any of that — it just automates fetching and running the
official Maestro release on AMD hardware.

## Credits

- **[Maestro](https://github.com/Blizaine/Maestro)**, by **Blizaine**, is
  the actual application. Everything creative you do in this app — every
  model, every mode, every feature described above — is Maestro's work.
  This repository is just an installer that points it at an AMD GPU
  instead of an NVIDIA one; all credit for the app itself belongs upstream.
- **[wan2gp-amd](https://github.com/6Morpheus6/wan2gp-amd)**, by
  **6Morpheus6**, is where the idea for *how* to do that porting came from.
  Its "clone the upstream app at install time instead of forking it"
  pattern for bringing a CUDA-only Pinokio app to AMD via ROCm is the
  approach this whole wrapper is built on.

If you find this useful, consider starring both of those projects — this
one exists because of their work.

---

## For developers

This repo is intentionally tiny: it's a Pinokio wrapper, not a fork. It
clones upstream [Blizaine/Maestro](https://github.com/Blizaine/Maestro)
fresh at install/update time rather than vendoring its source, following the
same model as [wan2gp-amd](https://github.com/6Morpheus6/wan2gp-amd).

- **`CLAUDE.md`** — full architecture notes: file layout, GPU-detection
  logic, how the ROCm wheel versions are pinned/bumped, and how the update
  flow preserves your downloaded models across upstream pulls. Start there
  before changing anything.
- **`launcher_profile.js`** — AMD GPU family detection.
- **`install.js` / `torch.js` / `start.js` / `update.js` / `start_latest.js`
  / `reset.js`** — the Pinokio script pipeline.
- This project is AMD-only by design; NVIDIA support lives in a separate
  project. PRs that add NVIDIA/CUDA branches here won't be merged.

Issues and PRs against the AMD wrapper itself (install flow, GPU detection,
ROCm wheel selection) are welcome. Bugs in Maestro's actual features belong
upstream at [Blizaine/Maestro](https://github.com/Blizaine/Maestro).
