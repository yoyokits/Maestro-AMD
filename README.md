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

### Proven settings

Before your first generation, read the
**[Maestro AMD Playbook](docs/SETTINGS.md)** — a known-good configuration measured on a real
RX 7900 XTX, with a start-small ladder to work up from. It covers the memory profile that
most often makes the difference between "works" and "appears frozen", and the settings behind
generations that actually completed.

**Start with the smallest setting in that ladder.** A stall on AMD looks exactly like a
crash, because nothing prints while it grinds — so prove the install works on a short,
low-resolution clip before asking for anything ambitious.

### Honest performance notes

- **Output quality is identical to the NVIDIA version.** Quality comes from
  the model weights, not the GPU brand — an AMD card and an NVIDIA card
  running the same model produce the same kind of results.
- **Speed is usually a bit behind an equivalent-tier NVIDIA card.** The
  NVIDIA build installs extra acceleration libraries — quantized attention
  (SageAttention), FlashAttention, and 4-bit inference kernels — that either
  have no ROCm build or aren't packaged for AMD on Windows. This wrapper
  uses PyTorch's own attention path instead. It works; it just isn't the
  fastest path that exists on NVIDIA. (Developers: `PERFORMANCE.md` has the
  full accounting and what's worth closing.)
- **Some GPUs can't do fast attention at all.** On RDNA 2 (RX 6000) and the
  Ryzen AI APUs, AMD's accelerated attention kernels may be missing
  entirely, which means long or high-resolution generations can fail with an
  out-of-memory error no matter how much VRAM you have. **Run Diagnose** —
  it tells you directly whether your card is affected. RX 7000/9000 series
  are unaffected.
- **"llama.cpp CUDA kernels unavailable" in the console is expected.** Those
  kernels are NVIDIA-only. GGUF models still work; the conversion step just
  runs on the CPU, so they load more slowly. Not a bug, and nothing to fix
  on your end.
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

### AMD-specific defaults are set for you

A few of Maestro's stock settings assume an NVIDIA GPU. This wrapper now
corrects them automatically on every Install, Update and Start — you don't
need to do anything. Listed here so you know what changed and why:

- **MiniMax H3 text encoder → GGUF Q4_K_M** (was *NVFP4 AWQ
  "(Recommended)"*). NVFP4 is an NVIDIA-only format and the "Recommended"
  label doesn't account for AMD. Left alone it could **hang your whole
  machine badly enough to need a hard reset.** That no longer happens, and
  you download the ~14.6 GB encoder you can use instead of the ~15.7 GB one
  you can't.

  Only the dangerous NVFP4 value is ever changed — if you pick a different
  encoder under **Advanced Settings → "H3 Text Encoder"**, your choice is
  kept. **GGUF Q2_K** (~8.5 GB) is the option if you're short on RAM.

  H3 is a heavy model either way — it's a large transformer with a 32B text
  encoder, and its sparse-attention mode needs Triton, which AMD doesn't
  ship on Windows, so it uses standard attention. It works; it's just not
  the quickest thing in Maestro. If an H3 generation seems stuck, check
  whether you've selected a **"Fused 4-Step (Experimental)"** variant —
  those are the ones that have caused trouble here.

- **Attention mode → `auto`** if it was ever set to Sage, Flash or
  xformers. None of those are installed on AMD, and `auto` picks the right
  backend for your hardware.

One setting still worth checking yourself: in **Settings → Services**, leave
**LLM Device** set to **CPU**. That's correct for everyone, not just AMD — it
keeps the Director planning LLM off your GPU so it doesn't compete with video
generation for VRAM.

Run **Diagnose** any time to confirm these are in place.

## Installing

1. Install [Pinokio](https://pinokio.computer) if you don't already have it.
2. Open the Maestro AMD listing on the Pinokio store — [pinokio.co/apps/github-com-yoyokits-maestro-amd](https://pinokio.co/apps/github-com-yoyokits-maestro-amd) — and install it from there, or point Pinokio at this repository directly (or search Discover for "Maestro AMD" once it's listed).
3. Click **Install**. This clones Maestro, sets up a Python environment,
   installs the AMD ROCm build of PyTorch for your specific GPU, and builds
   the web UI. Takes roughly 10–20 minutes depending on your connection —
   this does *not* download model weights yet, those come later, per model,
   the first time you use one.
7. Click **Start**. A browser tab opens with the Maestro UI once the server
   is ready.

### Keeping it updated

- **Start** — launch with what's currently installed.
- **Update & Start** — pull the latest Maestro and launch in one click. This
  is the easiest way to stay current; recommended as your everyday launch
  button.
- **Update** — just pull the latest Maestro without starting it.

### If something goes wrong

Try these in order — the first three all keep your downloaded models.

- **Diagnose** — prints a health report: which GPU was detected, whether
  PyTorch found it, whether fast attention is actually available, and what's
  missing. **Please include this output when reporting a problem.**
- **Roll back last update** — Maestro updates track upstream's latest
  version, so occasionally a new version breaks something. This returns you
  to the version you were on before your last Update. Only appears once
  you've run Update at least once.
- **Repair** — rebuilds the Python environment and the interface from
  scratch without touching your models, LoRAs, or generated outputs. This is
  the right fix for a broken or half-finished install.
- **Reset** — removes Maestro and the Python environment entirely (a clean
  slate). **Your downloaded models, LoRAs, and generated outputs live inside
  the folder Reset deletes**, so try Repair first, and back them up if you
  want to keep them.

### "Torch not compiled with CUDA enabled"

If Maestro crashes on start with this, look a few lines above it for:

```
[Runtime] PyTorch 2.14.0+cpu | CUDA none | CUDA unavailable
```

The `+cpu` is the problem — a dependency update replaced the AMD ROCm
version of PyTorch with a CPU-only one. **Click Update**; it now detects
this and reinstalls the correct version automatically. If that doesn't
work, click **Repair** (keeps your models).

Ignore the `Triton=missing`, `SageAttention=missing`, `FlashAttention is
unavailable` and `llama.cpp CUDA kernels unavailable` lines — those are
normal on AMD and appear on healthy installs too.

### A generation finishes but the video is solid grey

The clip saves, but it's a tiny file (~50 KB) and every frame is flat
grey. You may also see an *"AMD software detected that a driver timeout
has occurred"* popup near the end.

**If you're using MiniMax H3: check your text encoder first.** In
**Advanced Settings → "H3 Text Encoder"**, make sure it's **GGUF Q4_K_M**
(the default). The smaller **GGUF Q2_K** option is a low-RAM fallback and
has been seen to produce solid-grey output on AMD — the same result every
time, surviving reboots and rollbacks, because the encoder itself is
feeding the generator nonsense. Switch to Q4_K_M and try again.

**If it's already on Q4_K_M**, a model file may actually be damaged (a
download interrupted by a crash or cancelled job). Run **Repair**
(rebuilds the environment, keeps your models); if that doesn't clear it,
**Reset** and reinstall. The H3 files live in
`Maestro/app/ckpts/minimax_h3/` if you want to delete a suspect one by
hand and let Maestro re-fetch it.

**The "driver timeout" popup** is usually a *side effect* of the above,
not the cause: when your machine is low on RAM, the doomed final decode
crawls for many minutes — long enough for Windows' graphics watchdog to
reset the GPU. The **"Don't ask me again (Disables issue detection)"**
checkbox in that dialog only hides the message; it doesn't fix anything.
Leave it unchecked.

### A generation freezes (progress bar stuck, GPU idle, RAM at ~100%)

Your machine is out of memory and swapping to disk. H3 is a large model;
this app is tuned for 24 GB VRAM / 32 GB RAM and gets tight below that.

- **Free up RAM first.** Close other programs — web browsers especially.
  If you've had several failed generations in a row, **restart Maestro**
  (Stop, then Start) — its memory use grows with each failure. A reboot
  before a big H3 job is worth it. Keep **Settings → Services → LLM Device
  = CPU**.
- **Generate smaller, then work up.** Lower the resolution, use fewer
  frames or shorter clips, cut the number of reference images, leave
  **First Block Cache** on. Get one clean result, then raise settings a
  step at a time until it fails — that's your ceiling on this machine.
- Avoid the **"Fused 4-Step (Experimental)"** H3 variants, and don't queue
  several big H3 jobs in one session — the first generation after a fresh
  **Start** is the most reliable.
- The **Settings → System → "VAE Tiling"** dropdown has *no effect on H3*
  (H3 tiles its own decode already) — it only helps Wan, LTX and Hunyuan.

**Only for large generations that time out on a genuinely slow GPU** (not
for the grey-output case above): you can give Windows' graphics watchdog
more patience. In an **Administrator** Command Prompt:

```
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 60 /f
reg add "HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" /v TdrDdiDelay /t REG_DWORD /d 60 /f
```

then **reboot**. To undo, set both back to `2` and `5`, or delete them.
(Linux has no equivalent — this is Windows-only.)

### Optional extras

- **Install Inpaint Support** — adds SAM 3.1 segmentation, used by the
  experimental Inpaint mode. Installed separately because it takes several
  extra minutes and most people won't need it. It lives in its own isolated
  environment, so it can't disturb the main install. Experimental on AMD.

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
