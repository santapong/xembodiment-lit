# Practical recipe: fine-tuning an open VLA on a single custom low-cost arm

Scope: given one 6-DOF DIY arm (RoboLLM `hardware/`, Arduino Mega-class
firmware, single front camera), a laptop with **8 GB RAM and no dGPU**, and
fine-tuning budget limited to rented A100/4090 hours — which open VLA (or
cheap non-VLA baseline) to fine-tune, how much data it needs, what it costs,
and how to log data so a later evaluation note can score it. Every number
below is tagged with its source paper/doc; anything not found in a fetched
source is marked **not verified**.

## 1. Decision table

| Model | Params | Min *train* VRAM | Demos/task | Inference on CPU/8GB laptop? | Action rep | Ecosystem | arXiv |
|---|---|---|---|---|---|---|---|
| **SmolVLA** | 450M (100M action expert) | fits single consumer GPU; "6× less memory" than π0 during training | ~50 (SO-100 real-world sets: 50 per task) | Yes — designed for CPU/consumer-GPU deployment (paper's explicit design goal; no CPU-Hz number found — **not verified**) | Flow-matching, continuous, 50-step chunks | LeRobot native | 2506.01844 |
| **OpenVLA (base)** | 7B | 15 GB bf16 inference; full FT needs 8×A100; LoRA r=32 needs **59.7 GB** at batch 16 (still multi-GPU-class) | 10–150 (Franka fine-tune experiments) | No — 4-bit quant still needs 7 GB VRAM + a CUDA GPU, ~3 Hz on A5000 | Discrete action tokens | openvla/openvla, HF hub | 2406.09246 |
| **OpenVLA-OFT** | 7B base + 279M trainable (LoRA+head) | 8×A100/H100 80GB for the paper's runs (LoRA rank 32); no single-GPU minimum published — **not verified** | 20–300 (ALOHA tasks); 500/task suite (LIBERO) | No — GPU-only, throughput numbers are A100-measured | Continuous, L1-regression, parallel-decoded chunks | moojink/openvla-oft | 2502.19645 |
| **π0 / openpi** | 3.3B | LoRA ≥22.5 GB (RTX 4090 class); full FT ≥70 GB (A100/H100 80GB) | not published per-task in paper; openpi examples use tens-hundreds of episodes — **not verified exact count** | Inference-only ≥8 GB VRAM per openpi README; **no CPU path documented** | Flow-matching, continuous | Physical-Intelligence/openpi (JAX; PyTorch π0/π0.5 added, no mixed precision) | 2410.24164 |
| **GR00T N1 (2B)** | 2.2B (1.34B VLM) | pretrain: up to 1024×H100; post-training on **1×A6000** works if only adapters+DiT tuned (batch ≤200), vision-encoder tuning needs batch ≤16 | 30–300/task (paper's post-training sweep); 10% of a benchmark's demos still gets close to full-data Diffusion Policy | No — 63.9 ms/chunk on an L40 GPU is the fastest reported number, no CPU path | Flow-matching (DiT), continuous, embodiment-specific adapter MLPs | NVIDIA/Isaac-GR00T | 2503.14734 |
| **ACT** (baseline) | ~80M | single 11 GB RTX 2080 Ti, ~5 h/task | 50 (100 for one harder task) | Yes trivially — 0.01 s inference on the same 2080 Ti; no VLM, CPU-feasible | Continuous joint-position chunks + temporal ensembling | ALOHA / LeRobot | 2304.13705 |
| **Diffusion Policy** (baseline) | tens of millions (ResNet/U-Net or transformer) | single consumer GPU; DDIM inference 0.1 s on an RTX 3080 | 50–200/task (data-efficiency ablation covers 40–200) | Marginal — 0.1 s/step on a 3080-class GPU; CPU numbers **not verified/not published** | Diffusion over continuous action chunks | diffusion_policy / LeRobot | 2303.04137 |

## 2. Per-path detail

### SmolVLA (2506.01844)
- Architecture: SmolVLM-2 VLM backbone truncated to its first 16 layers, feeding a ~100M-parameter flow-matching action expert via interleaved cross-/self-attention; **450M params total** (also released as 0.24B and 2.25B variants) (2506.01844, p.10-11).
- Pretraining: 481 community HF datasets, **22.9k episodes / 10.6M frames**, trained 200k steps, batch 256, **~30k GPU-hours total pretraining** across 4 GPUs, but the paper states the model "can easily be trained on a single GPU due to its small size" and is "~40% faster to train and consumes 6× less memory" than π0 (2506.01844, p.5, p.10).
- Real-world fine-tune data: SO-100 pick-place/stacking/sorting datasets and one SO-101 pick-place dataset, each with **50 demonstrations per task** (10 trajectories × 5 start positions) (2506.01844, p.8).
- Real-world success rates (multi-task, SO-100): Pick-Place 75%, Stacking 90%, Sorting 70%, avg **78.3%**, beating single-task ACT (avg 48.3%) and multi-task π0-3.5B (avg 61.7%) despite being ~7× smaller (2506.01844, p.11, Table 3).
- Pretraining-on-community-data effect: without it, single-task SmolVLA averages 40%; with pretraining + multitask FT, 78.3% (2506.01844, p.12, Table 5) — the single largest lever in the paper.
- Async inference: decouples action execution from prediction/perception. On Pick-Place, async completes the task in 9.7s vs 13.75s sync (~30% faster) and finishes ~19 vs 9 pick cycles in a fixed 60s window, at comparable success rate (2506.01844, p.12, Fig. 5).
- CPU-only inference numbers: the paper states SmolVLA is "designed to be trained on a single GPU and deployed on consumer-grade GPUs or even CPUs" as a headline claim, but no Hz/latency number for a pure-CPU run is given in the fetched text — **not verified**.

### OpenVLA / OpenVLA-OFT (2406.09246, 2502.19645)
- Base OpenVLA: Llama-2 7B + fused DINOv2/SigLIP vision encoder, 970k OXE episodes pretraining, 21,500 A100-hours over 64 A100 GPUs / 14 days (2406.09246, p.6).
- Fine-tuning to a new Franka setup: full FT uses 8 A100s for 5–15 h/task, 10–150 demos; LoRA (rank 32) matches full-FT success while training only 1.4% of params, **59.7 GB VRAM at batch 16**, fine-tunes in **10–15 h on a single A100** (8× less compute than full FT) (2406.09246, p.10, Table 1).
- Quantization: bf16 inference 15 GB VRAM / ~6 Hz on an RTX 4090; 4-bit quantized inference drops to **7.0 GB VRAM** with success rate matching bf16 (71.9% vs 71.3% on BridgeData V2), running at 3 Hz on an A5000 — still requires a CUDA GPU, no CPU path documented (2406.09246, p.11, Table 2).
- OpenVLA-OFT recipe (parallel decoding + action chunking + continuous L1-regression actions) lifts LIBERO average success from 76.5% (base OpenVLA) to **97.1%**, and raises action-generation throughput **26×** (4.2 Hz → 109.7 Hz on an A100) (2502.19645, Table I/II).
- Ablated contributions on LIBERO: parallel decoding + action chunking alone: +14 pts absolute (76.5%→90.2%); adding continuous actions (L1 or diffusion): a further +5 pts absolute (2502.19645, Sec. V-B).
- Real ALOHA bimanual tasks use 20–300 demonstrations depending on task; OFT+ fine-tunes each task 50–150K steps on 8×A100/H100 80GB GPUs, chunk size K=25 (2502.19645, p.7-8, Table V). No single-consumer-GPU LoRA-fine-tune number is published for OFT — the LIBERO/ALOHA hyperparameter tables both specify 8×A100/H100 (Table IV/V) — **min single-GPU VRAM not verified**.

### π0 / openpi (2410.24164 + openpi README)
- π0: PaliGemma-3B VLM + flow-matching action expert, 3.3B total params, trained on ~10,000 h of cross-embodiment data (per SmolVLA's description of the π0 baseline, 2506.01844 p.10).
- From the openpi GitHub README (fetched): **LoRA fine-tuning needs >22.5 GB VRAM** (RTX 4090-class); **full fine-tuning needs >70 GB** (A100/H100 80GB-class); **inference alone needs >8 GB VRAM**. JAX is the original/complete implementation; a newer PyTorch port covers π0 and π0.5 but only supports full bf16 or full fp32 (no mixed precision), and does not yet support π0-FAST.
- Per-task demonstration counts for π0 fine-tuning are not given a specific number in the fetched README text — **not verified**; treat as "in the low hundreds," consistent with OpenVLA-OFT's comparable ALOHA task sizes (20–300).
- No CPU inference path is documented for π0/openpi — 8 GB VRAM is a GPU floor, not a CPU option.

### GR00T N1 (2503.14734)
- Dual-system VLA: NVIDIA Eagle-2 VLM (System 2, reasoning) + DiT flow-matching action head (System 1, motor control); released GR00T-N1-2B has **2.2B params total, 1.34B in the VLM**; inference for a 16-action chunk is **63.9 ms on an L40 GPU** (2503.14734, p.3).
- Pretraining: up to 1024 H100 GPUs, ~50,000 H100-GPU-hours for GR00T-N1-2B (2503.14734, p.8).
- Compute-constrained fine-tuning: explicitly tested on **a single A6000 GPU** — batch size up to 200 if only adapter layers (state/action encoders, action decoder) + DiT are tuned; batch size drops to ≤16 if the vision encoder is also tuned (2503.14734, p.8).
- 10%-data result: on real GR-1 humanoid tasks, GR00T-N1-2B trained on only **10% of teleop data** scores 42.6% average vs Diffusion Policy's 46.4% on the **full** dataset — i.e., GR00T N1 with 1/10th the data still lands within ~4 points of a strong baseline trained on everything (2503.14734, p.15, Table 3).
- New-embodiment adaptation: post-training uses per-embodiment MLP state/action encoders projecting to a shared DiT embedding space (2503.14734, p.3, "State and Action Encoders"); post-training freezes the VLM's language component and fine-tunes the rest, using 30/100/300 demos per task in the simulation sweep (2503.14734, p.14-15, Table 2).
- Dataset format: GR00T N1's training corpora build on the **LeRobot dataset format** directly (2503.14734, p.21, "Dataset Formats").

### ACT and Diffusion Policy (cheap non-VLA baselines)
- **ACT** (2304.13705): CVAE + Transformer, ~80M params, trained from scratch per task. Real ALOHA tasks use **50 demonstrations** (100 for one harder task, "Thread Velcro"), each demo ~8–14 s / 400–700 timesteps at 50 Hz control. Training: **~5 hours on a single 11 GB RTX 2080 Ti**; inference **~0.01 s** on the same GPU (2304.13705, p.6). Real success rates: 88–96% on the two headline tasks (Slide Ziploc, Slot Battery), 80-90% on several others. Action chunking is the dominant lever: success jumps from 1% (k=1, no chunking) to 44% (k=100) in the ablation (2304.13705, p.9, Fig. 8a).
- **Diffusion Policy** (2303.04137): CNN or transformer denoising-diffusion head over action chunks, receding-horizon control. Real-world Push-T uses **136 demonstrations**; other real tasks (pouring, spreading, bimanual egg-beater, shirt folding) use 50–284 demos. Data-efficiency ablation sweeps 40/60/90/130/200 demos and Diffusion Policy beats LSTM-GMM at every size (2303.04137, p.16, Fig. 15). Inference: DDIM with 100 train / 10 inference denoising steps gives **0.1 s latency on an Nvidia RTX 3080**; real deployments used up to 16 inference steps at similar latency. No CPU numbers found — **not verified**.

## 3. LeRobot dataset format essentials (v3.0, huggingface.co/docs/lerobot)

- Storage: tabular signals (state/action/timestamp) in **Apache Parquet**; camera frames concatenated per-episode and encoded to **MP4**, sharded per camera; schema/FPS/normalization stats live in `meta/info.json`, `meta/stats.json`, `meta/tasks.jsonl`, `meta/episodes/`.
- Standard sample keys returned by `LeRobotDataset[i]`: `observation.state`, `action`, one `observation.images.<camera_name>` tensor per camera, `timestamp`.
- Registering a new robot/embodiment is done by defining a **features dict** (per-key `dtype`, `shape`, `names`) at dataset-creation time — exactly what RoboLLM's own `lerobot_features()` helper does (see §4).
- Recording from a custom robot: `lerobot-record` (CLI) takes `--robot.type`, `--robot.cameras`, `--dataset.repo_id`, `--dataset.num_episodes`, `--dataset.single_task`; for a robot with no LeRobot driver plugin (like an Arduino-Mega arm), the supported path is the Python API — `LeRobotDataset.create(...)`, `dataset.add_frame(...)`, `dataset.save_episode()`, then **`dataset.finalize()` before `push_to_hub()`** (v3.0 doc, "Common Issues" — finalize is mandatory or the Parquet files are corrupt).
- `delta_timestamps` lets a `LeRobotDataset` return short temporal windows per key (e.g. `-0.2s..0s` of images) without re-recording — useful for adding observation history to a chunk-based policy after the fact.

## 4. Recommended path for this owner

Ties to `~/RoboLLM/ROADMAP.md` sim track and Tier items (repo path:
`~/RoboLLM/ROADMAP.md`):

- **A2 — `camera_logger` → LeRobot dataset format**: code-ready ("code
  ready; real recording waits for encoders", ROADMAP.md Tier A table).
  `~/RoboLLM/hardware/lerobot_logger.py` already implements a LeRobot v3
  features schema for this exact arm: `observation.images.front` (video,
  H×W×3), `observation.state` (float32, 7 = 6 joints + gripper),
  `action` (float32, same 7 axes), plus a repo-specific
  `observation.camera_lag_ms` sync-quality field
  (`~/RoboLLM/hardware/lerobot_logger.py:29-52`, `lerobot_features()`).
  This *is* "registering a new embodiment" per §3 — no further schema work
  needed once encoders land.
- **B1 — SmolVLA fine-tune, prep CPU-verified, paused on GPU**
  (ROADMAP.md: "MuJoCo scene → LeRobot dataset in sim (B1 preparation
  CPU-verified) → SmolVLA fine-tune (~450M, 3090/A100 hours) → break it
  deliberately and MEASURE the failure"). The owner has already committed
  to SmolVLA; the numbers above support that choice for this hardware
  profile specifically:
  - **8 GB RAM / no dGPU laptop** rules out training anything here.
    SmolVLA is the only model in the table explicitly designed to be
    "trained on a single GPU" and is 6× lighter in training memory than π0
    (2506.01844) — it is the cheapest rented-GPU-hour path of the VLA
    options, and a single rented 3090/4090/A100 hour is sufficient (unlike
    OpenVLA-OFT's 8×A100 hyperparameter tables, or π0 full-FT's ≥70 GB
    floor).
  - Community pretraining on SO-100/SO-101-class low-cost arms is a
    reasonable prior for RoboLLM's own DIY 6-DOF arm — same market
    segment (hobby servos, 5–6 V supply) as the SO-100/SO-101 arms
    SmolVLA is evaluated on (2506.01844, p.8, Fig. 4).
  - Demo count: SmolVLA's own real-world fine-tune sets use **50
    demonstrations per task** (2506.01844, p.8). That is the same order
    as ACT (50) and matches what `hardware/lerobot_logger.py --steps 150
    --fps 15` sessions would need to accumulate for a single pick-place
    task once encoders are wired.
  - `numpy` constraint: `~/RoboLLM/requirements-lerobot.txt` and
    `~/RoboLLM/hardware/README.md` ("LeRobot dataset recording") both flag
    that **LeRobot 0.6 needs NumPy 2.x** while ROS 2 Jazzy on this machine
    is pinned to **NumPy 1.26.4** — hence the separate `.venv-lerobot`
    venv the README already sets up. This applies unchanged to SmolVLA
    since it trains through the LeRobot library (2506.01844, p.10,
    "Implementation details": "We conduct our experiments using LeRobot").
  - `~/RoboLLM/requirements-smolvla.txt` is a compatibility shim
    (`-r requirements/smolvla.txt`); no direct pins were readable from that
    file alone, so verify the resolved package set inside `.venv-lerobot`
    before renting GPU time.
- **Baseline comparison to log alongside SmolVLA**: ACT is the cheapest
  possible sanity baseline (single 11 GB GPU, ~5 h, 50 demos,
  2304.13705) — worth running once with the same 50-episode
  `lerobot_logger.py` capture before spending rented-GPU hours on SmolVLA,
  since it gives a same-data comparison point almost for free and is what
  SmolVLA itself is benchmarked against (2506.01844, Table 3).
- If the arm later needs bimanual or very high-frequency control (outside
  current scope), OpenVLA-OFT's recipe (parallel decoding + action
  chunking + continuous L1 actions) is the proven upgrade path, but its
  published hyperparameters assume 8×A100/H100 — a different cost class
  than this project's rented-GPU-hour model.

## 5. Budget estimate

Assumption, stated explicitly: **$2.00/GPU-hour**, a mid-range cloud rate
for a single A100 40GB or RTX 4090 spot instance (vendor price, not from
any paper — label this as the one assumed number in this table).

| Path | GPU-hours (source) | Assumed $/hr | Estimated cost | Note |
|---|---|---|---|---|
| SmolVLA fine-tune, single task, ~50 demos | Not explicitly given per-task in the paper; the paper's own real-task fine-tune runs to 200k steps (2506.01844, p.10) but states the model trains fast on a single GPU due to its size — treat **4–8 h** as a planning estimate, **not verified against a paper-stated hour figure** | $2.00 | **$8–16** | Single most likely real cost for one task |
| ACT baseline, single task, 50 demos | **5 h** (2304.13705, p.6, "training takes around 5 hours on a single 11G RTX 2080 Ti GPU") | $2.00 (2080Ti-class is usually cheaper, ~$0.5-1/hr, but kept uniform here) | **$10** | Cheap sanity check before SmolVLA |
| OpenVLA LoRA fine-tune, single task | **10–15 h on 1×A100** (2406.09246, p.10) | $2.00 | **$20–30** | Only relevant if pivoting off SmolVLA |
| OpenVLA-OFT fine-tune, ALOHA-scale task | 50-150K steps on **8×A100/H100**, hours not stated exactly — **not verified**, order-of-magnitude only | $2.00 × 8 GPUs | not estimable without an hours figure | Out of budget class for this project |
| π0 LoRA fine-tune | VRAM known (>22.5 GB) but no GPU-hours figure in the fetched README — **not verified** | $2.00 | not estimable | — |
| GR00T N1 post-training, adapter-only, single A6000 | Not stated in hours; batch-size ceiling given, not wall-clock — **not verified** | $2.00 | not estimable | — |

Only SmolVLA and ACT have costs groundable enough (even loosely) to
budget against; the other three require either a stated GPU-hour figure
this fetch did not surface, or a different cost class (multi-GPU) than
this project uses.

## 6. Break-and-measure: what to log

(Evaluation methodology itself is a sibling note — this is the minimum
capture list so that note has something to score.)

- Per-episode: full `observation.state`, `action`, and
  `observation.camera_lag_ms` streams already captured by
  `lerobot_logger.py` (§4) — keep these even for "failed" episodes,
  don't discard.
- Per-rollout outcome: binary success/fail plus **where** it failed
  (grasp / transport / placement) — SmolVLA's own real-world eval uses
  exactly this decomposition (0.5 for grasp, 0.5 for placement,
  2506.01844 p.8) and it is cheap to replicate by hand-labeling video.
  ACT's rubric-based partial scoring on ALOHA tasks is the harder-task
  precedent for the same idea (2304.13705, p.8).
  GR00T N1's real-robot rubric similarly reports partial per-phase scores.
- Timing: wall-clock time-to-completion per episode (SmolVLA's
  sync-vs-async comparison logs exactly this, 2506.01844 p.12) — needed
  if async inference is ever tried on the laptop-adjacent hardware.
  Frequency at which the policy was actually queried (Hz), since
  inference throughput differs sharply across models in §1/§2.
  Distinguish "commanded" vs "measured" state at logging time —
  `lerobot_logger.py`'s `--allow-commanded-state` flag exists exactly to
  prevent silently mislabeling open-loop targets as ground-truth
  measurements (`~/RoboLLM/hardware/README.md`, "LeRobot dataset
  recording").
- Out-of-distribution probe episodes (object moved to an unseen
  position) alongside in-distribution ones — every paper surveyed here
  reports an ID/OOD split (SmolVLA Table 4, GR00T N1 Table 5) and it is
  the cheapest way to get a generalization number, not just a fit number.

## 6b. Data-strategy pointers (added 2026-09-03, not verified)

Three papers on where robot data comes from arrived via an alphaXiv Assistant transcript and are
filed in the alphaXiv folder *Data & simulation*. Ids confirmed to resolve; **contents not read**,
so nothing here is a sourced claim.

| Paper | arXiv | Why it is relevant to this recipe |
|---|---|---|
| Data Pyramid for Embodied Manipulation: A Survey | 2607.24744 | the layered strategy this note already applies informally: cheap broad data under a small high-quality task set. Read it before scaling the demo count |
| Data Standards for Humanoid Robotics | 2606.19769 | whether physical experience can accumulate across robots and orgs at all |
| 3D Generation for Embodied AI and Robotic Simulation | 2604.26509 | synthetic scene generation as a substitute for collecting more real demos |

The augmentation result in `world-models.md` §3 is the one *verified* data point nearby, and it is
narrow: style-transferred synthetic data bought robustness to held-out backgrounds and lighting, at
a small in-distribution cost, using a 38B generator far outside this note's budget (2607.11643).
Treat generated data as a robustness tool, not a way to collect fewer demonstrations.

## 7. Limitations

- No paper in this set publishes a single, apples-to-apples GPU-hour
  figure for "fine-tune model X on N demos of a brand-new low-cost arm" —
  every hour figure here comes from a different task/hardware context
  (ALOHA, LIBERO, Franka, GR-1) and is being used as an order-of-magnitude
  proxy, not a guarantee.
- The $2.00/GPU-hour budget assumption in §5 is not sourced from any paper
  or vendor page fetched in this session; it is a placeholder the owner
  should replace with an actual quoted rate before committing rented-GPU
  time.
- No CPU-only inference benchmark (Hz, latency) was found in the fetched
  text for SmolVLA, π0/openpi, or Diffusion Policy, despite SmolVLA's
  paper claiming CPU deployability as a design goal — treat "runs on the
  laptop" as unverified until measured locally.
- OpenVLA-OFT and π0/openpi hyperparameter tables are both multi-A100/H100
  by default; no minimum single-consumer-GPU LoRA configuration is
  published for OFT, so its entry in §1 for "min train VRAM" is an
  extrapolation from OpenVLA-base's LoRA number (2406.09246), not a
  number OFT's own paper states.
- Per-task demonstration counts for π0 fine-tuning were not found in the
  openpi README fetch; the "not verified" figure in §1/§2 should be
  replaced once the openpi examples/config directory is read directly
  (out of scope for this note, which used the top-level README only).
- `~/RoboLLM/requirements-smolvla.txt` and `requirements-lerobot.txt` are
  both thin `-r requirements/...` compatibility shims; this note did not
  resolve the underlying pinned versions, so the exact SmolVLA/LeRobot
  version pair the owner will train against is not confirmed here.

## 8. Sources

- SmolVLA — arXiv:2506.01844
- OpenVLA — arXiv:2406.09246
- OpenVLA-OFT (Fine-Tuning Vision-Language-Action Models) — arXiv:2502.19645
- π0 (Physical Intelligence) — arXiv:2410.24164; openpi README,
  https://github.com/Physical-Intelligence/openpi (fetched via raw README)
- GR00T N1 — arXiv:2503.14734
- ACT / ALOHA — arXiv:2304.13705
- Diffusion Policy — arXiv:2303.04137
- LeRobot dataset v3.0 docs — https://huggingface.co/docs/lerobot/lerobot-dataset-v3
- `~/RoboLLM/ROADMAP.md`
- `~/RoboLLM/hardware/README.md`
- `~/RoboLLM/hardware/lerobot_logger.py`
- `~/RoboLLM/hardware/camera_logger.py`
- `~/RoboLLM/requirements-smolvla.txt`, `~/RoboLLM/requirements-lerobot.txt`
