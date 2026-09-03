# Frontier VLA Models, 2025–2026 (post-GR00T N1 / CogACT)

Updated: 2026-08-26

Extends `survey.md` §3's lineage: RT-1 → RT-2 → RT-2-X → OpenVLA → π0 → CogACT →
{SpatialVLA / X-VLA / Gemini Robotics / Helix}. This note picks up from the 10 anchors
already in the repo (RT-1/2, RT-X, Octo, OpenVLA, Diffusion Policy, π0, DROID, GR00T N1,
CogACT) and covers what came after.

## 1. Extended lineage

```
π0 (2410.24164) ──┬─→ π0-FAST (2501.09747, tokenization swap)
                   ├─→ π0.5 (2504.16054, co-training + hierarchical inference)
                   ├─→ π0-EqM (2605.23128, equilibrium-matching decoder swap)
                   └─→ [2026-wave descendants below]

OpenVLA (2406.09246) ──→ OpenVLA-OFT (2502.19645, fine-tuning recipe, not a new base model)

RDT-1B (2410.07864) ──→ X-VLA (2510.10274, soft-prompt cross-embodiment successor)

GR00T N1 (2503.14734) ──→ GR00T N1.5 / N1.6 / N1.7 (NVIDIA GitHub/HF, no arXiv paper for
                           N1.5/N1.6; N1.7 arXiv page is the same 2503.14734 landing page)

CogACT (2411.19650) ──→ GR-3 (2507.15493, ByteDance, dual-arm-mobile "ByteMini")

Gemini Robotics (2503.20020) — new branch: closed VLM-first (Gemini 2.0) architecture,
                                distinct from the Physical-Intelligence/OpenVLA lineage.

SmolVLA (2506.01844) — new branch: community-data, sub-1B efficiency-first VLA (LeRobot).
```

## 2. Anchor models

### π0-FAST — arXiv 2501.09747
- **Architecture**: keeps π0's autoregressive VLM path but replaces per-dimension binning
  tokenization with **FAST** — discrete cosine transform (DCT) of the normalized action
  chunk, quantized, then compressed with byte-pair encoding (BPE) into a dense token
  sequence (2501.09747).
- **Action representation**: FAST consistently produces "roughly 30 action tokens per
  chunk per robot arm" independent of control frequency, vs. up to 700 tokens/chunk for
  naive binning on 50 Hz bimanual T-shirt folding (Table I, 2501.09747).
- **Data**: FAST+ universal tokenizer trained on **1M real robot action trajectories**;
  π0-FAST generalist policy trained on the π0 cross-embodiment mixture, "903M timesteps"
  from PI's own datasets plus 9.1% from BridgeV2/DROID/OXE (2501.09747).
- **Headline numbers**: matches diffusion π0 performance on dexterous long-horizon tasks
  (laundry folding, table bussing) while training with **up to 5× fewer GPU hours**
  (2501.09747, Fig. 1, Fig. 11). Inference is slower than diffusion π0 (~750 ms/chunk vs.
  ~100 ms/chunk on an RTX 4090) because it needs 30–60 autoregressive decode steps vs. 10
  diffusion steps (2501.09747, §VI-E).
- **What it changed**: first "zero-shot" language-conditioned generalist policy evaluated
  on DROID in a completely unseen environment without co-training/fine-tuning (2501.09747).
  Showed autoregressive-with-good-tokenization can match flow-matching quality at a
  fraction of training compute.

### π0.5 — arXiv 2504.16054
- **Architecture**: two-stage training on top of π0's PaliGemma (SigLIP 400M + Gemma 2B)
  VLM backbone plus a 300M-parameter flow-matching "action expert." Pre-training uses
  discrete FAST tokens for all modalities (including actions); post-training adds the
  flow-matching action expert for continuous, fast inference (2504.16054, Fig. 3).
- **Action representation**: hierarchical — the model first autoregressively predicts a
  high-level subtask string (e.g. "pick up the pillow"), then flow-matches a low-level
  continuous action chunk (horizon H=49) conditioned on that subtask (2504.16054, §IV-A).
- **Data**: ~400 hours of mobile-manipulation teleop data is only **2.4%** of the training
  mixture; the rest is cross-embodiment robot data, lab data, web multimodal data
  (captioning/QA/localization), and verbal instruction data (2504.16054).
- **Headline numbers**: performs 10–15 minute long-horizon multi-stage cleaning tasks
  (kitchens, bedrooms) in **entirely new homes never seen in training**; outperforms both
  π0 and an enhanced "π0-FAST+Flow" baseline trained on the same robot data (2504.16054,
  Fig. 12, §V-D). Ablations show removing verbal-instruction or web data significantly
  degrades high-level task inference (2504.16054, Fig. 13).
- **What it changed**: first demonstration that heterogeneous co-training (97.6% non-robot
  or non-target-embodiment data) drives open-world generalization to unseen environments,
  not just unseen objects within a known scene (2504.16054).

### OpenVLA-OFT — arXiv 2502.19645
- **What it is**: a fine-tuning *recipe*, not a new pretrained base model — OpenVLA-OFT
  keeps OpenVLA's Llama-2 7B + DINOv2/SigLIP backbone but changes how it's adapted.
- **Architecture changes**: (1) **parallel decoding** — bidirectional attention replaces
  causal, predicting all action tokens in one forward pass instead of autoregressively;
  (2) **action chunking** (K=8 or K=25 steps); (3) **continuous actions** via a 4-layer MLP
  head trained with L1 regression instead of discrete binned tokens; (4) optional FiLM
  language-conditioning for the "+" variant (2502.19645, §IV).
- **Headline numbers**: raises OpenVLA's LIBERO average success rate from **76.5% → 97.1%**
  while giving a **26× throughput speedup** (8-step chunks) — up to **43× on real ALOHA
  hardware** with 25-step chunks (2502.19645, Table I, Table II). On real bimanual ALOHA
  tasks, OpenVLA-OFT+ beats fine-tuned π0 and RDT-1B and from-scratch Diffusion
  Policy/ACT by up to 15 points absolute (2502.19645, Abstract).
- **What it changed**: showed that fine-tuning design choices (decoding scheme, action
  representation, objective) matter as much as base-model architecture — a 7.5B-parameter
  autoregressive model with the right recipe can match or beat larger diffusion/flow VLAs
  on speed and success rate simultaneously (2502.19645).

### SmolVLA — arXiv 2506.01844
- **Architecture**: SmolVLM-2 vision-language backbone (SigLIP + SmolLM2), **truncated to
  its first half of layers** (N=L/2) to save compute, feeding an action-expert transformer
  with **interleaved cross-attention/self-attention blocks** trained via flow matching
  (2506.01844, §3.1).
- **Size**: main model is **450M total parameters**, ~100M in the action expert — an order
  of magnitude smaller than π0 (3.3B) or OpenVLA (7B) (2506.01844, §4.3).
- **Data**: pretrained on **fewer than 30k episodes** (~23k trajectories) from public,
  community-collected LeRobot datasets on a single low-cost robot (SO100), not
  industrial/academic fleets — "an order of magnitude less data than prior art"
  (2506.01844, Introduction, Limitations).
- **Headline numbers**: on LIBERO, SmolVLA-0.45B scores **87.3%** average, beating
  VLM-only-initialized π0-Paligemma-3B (71.8%) and close to robotics-pretrained π0-3.3B
  (86.0%) (2506.01844, Table 2). On real-world SO100 tasks it beats π0-3.5B (78.3% vs.
  61.7% average) despite being ~7× smaller (2506.01844, Table 3). Training is ~40% faster
  and uses 6× less memory than π0 (2506.01844, §4.5). Introduces **asynchronous inference**
  (decoupling perception/prediction from execution) giving ~30% faster task completion at
  matched success rate (2506.01844, §4.6, Fig. 5).
- **What it changed**: proof that community-scale data + a sub-1B model can rival
  billion-parameter VLAs pretrained on proprietary fleets, at consumer-GPU/CPU deployment
  cost.

### Gemini Robotics — arXiv 2503.20020
- **Architecture**: built on **Gemini 2.0**, split into two models. **Gemini Robotics-ER**
  is a VLM with enhanced "embodied reasoning" (object detection, pointing, trajectory/grasp
  prediction, 3D bounding boxes and multi-view correspondence) usable zero-/few-shot without
  any action fine-tuning. **Gemini Robotics** itself is a distilled VLA: a cloud-hosted "VLA
  backbone" (distilled Gemini Robotics-ER) plus a local low-latency "action decoder" running
  on the robot's onboard computer (2503.20020, Fig. 14, §3.1).
- **Action representation / control**: backbone query-to-response latency reduced from
  seconds to **under 160 ms**; combined end-to-end latency from raw observation to action
  chunk is **~250 ms**, giving an effective **50 Hz** control frequency via chunked actions
  (2503.20020, §3.1).
- **Data**: "thousands of hours" of teleoperated ALOHA 2 demonstrations collected over 12
  months, co-trained with web documents, code, multimodal (image/audio/video) content, and
  embodied-reasoning/VQA data (2503.20020, §3.1).
- **Headline numbers**: introduces the **ERQA benchmark** (400 embodied-reasoning VQA
  questions) where Gemini 2.0 is state-of-the-art among VLMs (2503.20020, §2.1). In
  zero-shot code-generation robot control, Gemini Robotics-ER nearly doubles task
  completion vs. vanilla Gemini 2.0 Flash (2503.20020, §2.3). Specialized fine-tuned
  variants solve origami folding, card games, and adapt to new embodiments (bi-arm
  platform, high-DoF humanoid) from as few as 100 demonstrations (2503.20020, Abstract).
- **What it changed**: the first frontier closed general-purpose VLM (not a
  robotics-specialist backbone) shown to transfer directly into a competitive VLA, with
  embodied reasoning as a reusable, separately-releasable capability (Gemini Robotics-ER)
  distinct from low-level control.

### RDT-1B — arXiv 2410.07864 (ICLR 2025)
- **Architecture**: **Robotics Diffusion Transformer**, a Diffusion Transformer (DiT)
  backbone scaled to **1.2B parameters** — "the largest diffusion-based foundation model for
  robotic manipulation" at time of release — with QKNorm/RMSNorm for training stability, an
  MLP decoder instead of a linear head, and alternating cross-attention condition injection
  for image/language conditioning (2410.07864, §4.1, Fig. 3).
- **Action representation**: diffusion denoising over a **Physically Interpretable Unified
  Action Space** that maps heterogeneous single-arm/dual-arm/mobile-base action spaces into
  one physically-grounded format for cross-robot pretraining (2410.07864, §4.2).
- **Data**: pretrained on the largest multi-robot dataset collection to date at the time (a
  "1M episode" collection of 46 datasets incl. Open X-Embodiment), fine-tuned on a
  self-collected 6K+ episode bimanual dataset (2410.07864, Fig. 1).
- **Headline numbers**: **56% average success-rate improvement** over baselines (ACT,
  OpenVLA, Octo) across 7 challenging bimanual tasks; zero-shot generalization to unseen
  cups/rooms and 1–5-shot learning of new skills (handover, folding) where baselines score
  near 0% (2410.07864, Table 3, Abstract).
- **What it changed**: showed diffusion-based (not autoregressive-token or flow-matching)
  action generation scales to 1B+ parameters for bimanual manipulation specifically, and
  established the unified-action-space technique later echoed by X-VLA/GR00T-style
  cross-embodiment training.

### GR-3 — arXiv 2507.15493 (ByteDance Seed, Jul 2025)
- **Architecture**: mixture-of-transformers — Qwen2.5-VL-3B-Instruct VLM backbone (frozen
  co-trained) plus an action diffusion transformer (DiT) with **half as many layers as the
  VLM**, using flow matching for action prediction; **4B parameters total**. Extra
  RMSNorm after DiT linear layers was found critical for training stability and
  instruction-following (2507.15493, §2).
- **Action representation**: k-length flow-matched action chunk plus an auxiliary
  "task status" dimension (Ongoing/Terminated/Invalid) used to force the model to attend to
  language rather than exploit visual shortcuts (2507.15493, §3.1).
- **Data**: three co-trained sources — robot teleop trajectories (via a data-collection
  scheduler for diversity), web-scale vision-language data (captioning/VQA/grounding), and
  **human trajectories collected via VR (PICO 4 Ultra)** at ~450 trajectories/hour vs. ~250
  for teleoperated robot data (2507.15493, §3).
- **Hardware**: introduces **ByteMini**, a 22-DoF bi-manual mobile robot with a compact
  sphere-wrist 7-DoF arm design (2507.15493, §4.1).
- **Headline numbers**: beats π0 across all evaluated real-world tasks. On
  Unseen-Instructions/Unseen-Objects generalization, boosts success rate from **40% (π0)
  to 77.1% / 57.8%** (2507.15493, §5.1). With only **10 human VR trajectories per object**,
  boosts unseen-object success further to **86.7%** (2507.15493, §5.1). On long-horizon
  table bussing with instruction-following, GR-3 hits **97.5% vs. π0's 53.8%**
  (2507.15493, §5.2). On dexterous cloth-hanging, 86.7%/83.9%/75.8% task progress across
  Basic/Position/Unseen-clothes settings (2507.15493, §5.3).
- **What it changed**: demonstrated VR-collected human trajectories (not robot teleop) as
  an efficient few-shot adaptation channel, and pushed co-training with VL data specifically
  for *abstract-concept* instruction following (sizes, spatial relationships) beyond
  π0.5's household-cleaning scope.

### X-VLA — arXiv 2510.10274 (Oct 2025)
- **Architecture**: "Soft-Prompted Transformer" — a **0.9B**-parameter flow-matching VLA
  using only standard self-attention Transformer encoder blocks (no DiT, no MM-DiT). Each
  data source (robot platform/camera config) gets its own set of learnable **soft-prompt**
  embeddings injected early, rather than only a separate output action-decoder head as prior
  work (RDT/GR00T/π0) does (2510.10274, Fig. 1, §3).
- **Action representation**: flow-matching over action chunks; domain-specific input/output
  linear projections plus soft prompts absorb embodiment heterogeneity (only ~0.04% of
  parameters are domain-specific) (2510.10274, Fig. 10).
- **Data**: 290K episodes from DROID, RoboMIND, and AgiBot across **seven platforms /
  five arm types** (2510.10274, §1).
- **Headline numbers**: SOTA on 5 of 6 simulation benchmarks — **Simpler-WidowX 80.4%,
  Libero 98.1%, Calvin, RoboTwin-2.0 (Easy 70.0/Hard 39.0)** — beating GR00T-N1 (93.9% on
  LIBERO avg), π0 (94.1%), OpenVLA-OFT (97.1%) (2510.10274, Table 2). With **LoRA tuning
  only 9M params (~1% of the model)**, reaches 93% LIBERO / 54% Simpler-WidowX, comparable
  to fully fine-tuned π0 (3B params) — a **300× parameter-efficiency** claim for adaptation
  (2510.10274, Abstract, §5.2).
- **What it changed**: moved cross-embodiment heterogeneity handling earlier in the
  pipeline (soft prompts at the input/feature level) rather than only at the action-decoder
  head, and showed a much smaller (0.9B) model can match or beat billion-parameter VLAs
  through better heterogeneity handling rather than raw scale.

### GR00T N1.5 / N1.6 / N1.7 — NVIDIA GitHub/HuggingFace (no separate arXiv paper)
- **Status**: **non-peer-reviewed / blog+repo only.** The only arXiv paper is GR00T N1
  itself (2503.14734), already in the repo. N1.5/N1.6/N1.7 are documented solely via the
  NVIDIA/Isaac-GR00T GitHub repo and HuggingFace model cards — flagged per the sourcing
  rules.
- What was fetched (GitHub repo text, not independently verified against a paper): N1.7
  changes the VLM backbone from "Eagle" to **Cosmos-Reason2-2B** (Qwen3-VL architecture),
  reduces diffusion layers **32→16**, expands state/action dimensions **29→132**, expands
  action horizon **16→40**, and adds a relative end-effector action space; stated as a
  **3B-parameter** model. No quantitative before/after benchmark numbers were found in the
  fetched content — **not verified**.
- Third-party papers in this note's own citation set report GR00T-N1.5 numbers on
  LIBERO/SimplerEnv/RoboCasa/LIBERO-Plus (e.g. LIBERO avg 86.5% per FoMoVLA Table 1;
  RoboCasa GR-1 Tabletop 47.6% for "GR00T-N1.6" per FoMoVLA Table 3; LIBERO-Plus 59.0% per
  NebulaVLA Fig. 3) — these are secondary citations from other papers' baseline tables, not
  numbers from an NVIDIA GR00T N1.5/N1.6 paper, and are reproduced here only as
  third-party-reported baseline figures, flagged accordingly.

### Figure Helix — figure.ai blog (non-peer-reviewed / blog only)
- **Status**: blog announcement only, **no arXiv paper**, confirmed via WebFetch of
  figure.ai/news/helix.
- **Architecture** (per blog): dual "System 1 / System 2" design. **System 2 (S2)** is a
  7B-parameter open-weight VLM operating at **7–9 Hz** for scene understanding. **System 1
  (S1)** is an 80M-parameter reactive visuomotor transformer running at **200 Hz**.
- **Action representation**: continuous, untokenized full upper-body humanoid control (35
  DoF) — desired wrist poses, finger flexion/abduction, torso and head orientation targets.
- **Data**: ~500 hours of teleoperated data total, stated as under 5% of prior VLA dataset
  sizes; claims zero-shot generalization to "thousands of novel items."
- **Verification note**: no numerical benchmark comparisons are given in the blog; all
  numbers above are as-stated by Figure AI's own post and are **not independently
  verified**.

## 3. 2026 wave (short entries)

Six papers were requested by exact 26XX.NNNNN-style arXiv id. All six IDs resolved
successfully via alphaXiv (the 2607–2608 numbering did fetch, contrary to the possibility
flagged in the brief) and PDF content was retrieved directly, so entries below are backed
by primary source text, not "not verified" placeholders.

| Model | arXiv | One-line result |
|---|---|---|
| **Pelican-VLA 0.5** | 2607.06655 | Unified VLM+future-frame+action model on Qwen3-VL 4B with learnable "BotTokens" bottleneck between perception and action; shows zero-shot attention already localizes on manipulation-relevant objects/contact regions before fine-tuning. After RoboTwin fine-tuning: **91.4% (Clean) / 91.0% (Randomized)** avg success, best among compared open-source VLAs (2607.06655, Table 1). |
| **FoMoVLA** | 2607.14739 | Adds future-feature prediction (MAE-style bottleneck tokens) + sparse 2D point tracking as training-only auxiliary objectives, coupled via a future-conditioned cross-attention module; zero added inference cost. LIBERO avg **98.8%** (full model), RoboCasa GR-1 Tabletop **56.9%** vs. 47.8% base backbone (+9.1 pts), LIBERO-Plus OOD **80.5%** (2607.14739, Tables 1–3). |
| **HAF** | 2608.16837 | Adapts generalist flow-matching VLAs (π0.5-based) to **humanoid whole-body loco-manipulation** via hierarchical 3-stage action-flow generation (locomotion→waist→arms with cross-stage KV-cache) plus HAF-Steer, a DCT-compressed latent-noise RL (SAC) refinement stage. Raises avg normalized real-world task score from π0.5's **53.3% to 70.5%** across 7 humanoid tasks (2608.16837, §5.2). |
| **NebulaVLA** | 2608.16503 | Dual-frequency (System2 10 Hz Qwen3-VL planner / System1 20 Hz DiT controller) VLA with GESTURE-7, a 7D natural-language-keyword end-effector action representation for cross-embodiment unification, and a "Guide Action" mechanism (image-outpainting-style masked denoising) to remove chunk-boundary jitter. **85.5%** avg success on LIBERO-Plus (vs. GR00T-N1.5 59.0%, π0.5 58.0% as reported in this paper); ~2.7× faster action generation; real-world jerk reduced 25.6% via Guide Action (2608.16503, Fig. 3, Table 3). |
| **π0-EqM** | 2605.23128 | Swaps π0's flow-matching action decoder for an Equilibrium Matching (EqM) decoder — a time-invariant vector field solved iteratively to a stationary point rather than integrated over fixed diffusion/flow timesteps — enabling adaptive stopping depth and warm-starts across control cycles. Under matched 300-step budget: RoboTwin avg success **40.4%→50.2%** (19 tasks), LIBERO-10 **85.2%→87.0%** (2605.23128, Tables I–II). Identifies a task-dependent "stationarity–executability gap" where lower residual doesn't monotonically mean better execution. |
| **Reflective VLA** | 2606.25215 | Conditions each action prediction on a rolling context of past observation–action–consequence triplets (not just observation history) via a block-causal-masked dual-system VLA, letting the policy infer embodiment-specific latent factors (camera calibration, actuation bias) from interaction evidence rather than a single frame. LIBERO avg **97.6%** (SOTA), SimplerEnv-Bridge **78.2%** (SOTA); under camera/robot-calibration distribution shift (LIBERO-Plus-Hard), **+4.2 pts** over a matched reactive π0.5 baseline (68.8% vs 64.6%), with the largest single-ablation gain (73.1%→77.8%) coming specifically from including the action-aligned consequence observation, not just longer context (2606.25215, Tables 1–3). |

## 4. Updated Quick-Facts table

| Model | Params | Action rep | Data | Headline result | arXiv |
|---|---|---|---|---|---|
| π0-FAST | 2B (VLM) + FAST tokenizer | DCT+BPE discrete tokens, autoregressive | 903M timesteps (PI mixture) + 1M-traj universal tokenizer | Matches diffusion π0 quality, 5× less training compute | 2501.09747 |
| π0.5 | 2B VLM + 300M action expert | Hybrid: discrete (pretrain) → flow-matching (post-train), hierarchical subtask→action | ~400h mobile-manip + heterogeneous co-training (97.6% non-target data) | Long-horizon (10–15 min) cleaning in unseen homes | 2504.16054 |
| OpenVLA-OFT | 7.5B (OpenVLA base) | Continuous, parallel-decoded, L1-regression, chunked | Same as OpenVLA (970k OXE) + task fine-tune sets | LIBERO 76.5%→97.1%, 26–43× throughput | 2502.19645 |
| SmolVLA | 0.24B–2.25B (main: 0.45B) | Flow-matching, interleaved cross/self-attn | <30k community episodes (LeRobot) | Matches/beats π0-3.3B at ~7× fewer params | 2506.01844 |
| Gemini Robotics | not disclosed (Gemini 2.0-based) | Flow-matched action chunks via cloud backbone + local decoder | 1000s hrs ALOHA 2 teleop + web multimodal | ~50 Hz control, new embodiments from 100 demos | 2503.20020 |
| RDT-1B | 1.2B | Diffusion (DiT) over unified action space | 1M-episode multi-robot pretrain + 6K bimanual fine-tune | +56% success over ACT/OpenVLA/Octo baselines | 2410.07864 |
| GR-3 | 4B | Flow-matching DiT + VLM (Qwen2.5-VL-3B) | Robot teleop + web VL data + VR human trajectories | Unseen-object success 40%→86.7% (π0→GR-3+10-shot VR) | 2507.15493 |
| X-VLA | 0.9B | Flow-matching, soft-prompted Transformer encoder | 290K episodes, 7 platforms | SOTA on 5/6 sim benchmarks; 300× cheaper PEFT adaptation | 2510.10274 |
| GR00T N1.5/N1.6/N1.7 | ~3B | Flow-matching diffusion (16–32 layers) | not verified (blog-documented only) | not verified (no arXiv benchmark numbers found) | none (blog/GitHub only) |
| Figure Helix | S2 7B + S1 80M | Continuous, untokenized, dual-frequency (7–9 Hz / 200 Hz) | ~500h teleop | Zero-shot generalization to "thousands of novel items" (unverified) | none (blog only) |
| Pelican-VLA 0.5 | 4B (Qwen3-VL) | Flow-matching, BotTokens bottleneck | ~2400h heterogeneous (RoboTwin fine-tune) | RoboTwin 91.4%/91.0% (Clean/Randomized) | 2607.06655 |
| FoMoVLA | on StarVLA-GR00T backbone | Flow-matching + future-feature/point-track aux losses | LIBERO, RoboCasa GR-1 | LIBERO 98.8%, RoboCasa +9.1 pts over backbone | 2607.14739 |
| HAF | π0.5-based, humanoid | Hierarchical 3-stage flow-matching + DCT-RL steering | 7 real humanoid loco-manip tasks | Real-task score 53.3%→70.5% vs. π0.5 | 2608.16837 |
| NebulaVLA | Qwen3-VL-based S2 + DiT S1 | GESTURE-7 (7D NL-keyword end-effector) | LIBERO-Plus, AgiBot A2 real-world | LIBERO-Plus 85.5%, ~2.7× faster inference | 2608.16503 |
| π0-EqM | π0-based (decoder swap only) | Equilibrium Matching (time-free iterative solve) | RoboTwin (19 tasks), LIBERO | RoboTwin 40.4%→50.2% at matched compute | 2605.23128 |
| Reflective VLA | dual-system VLA (π0.5-style) | Flow-matching + in-context (O,A,O') triplets | LIBERO, SimplerEnv-Bridge, LIBERO-Plus(-Hard) | LIBERO-Plus-Hard 64.6%→68.8% under distribution shift | 2606.25215 |

## 5. Themes that changed since 2024

1. **Fine-tuning recipe is now a first-class research object, not an afterthought.**
   OpenVLA-OFT (2502.19645) shows the exact same base model can go from 76.5%→97.1% on
   LIBERO purely from decoding/action-representation/objective choices — no new
   pretraining. This complicates any claim that architecture X "beats" architecture Y
   without controlling for fine-tuning recipe.
2. **Action decoder is becoming a swappable module.** π0-EqM (2605.23128) and X-VLA's
   backbone ablation (Table 4, 2510.10274) both treat the flow-matching/diffusion decoder
   as interchangeable with the rest of the VLA stack held fixed — a shift from "new
   architecture" papers to "new decoder for existing stack" papers.
3. **Cross-embodiment heterogeneity handling moved earlier in the pipeline.** RDT's unified
   action space (output-side) → X-VLA's soft prompts (input/feature-side, 2510.10274) is a
   trend toward absorbing embodiment differences before the shared backbone rather than
   only at the action head.
4. **Co-training data mix, not parameter count, is now the main lever for
   generalization.** π0.5 (2504.16054), GR-3 (2507.15493), and SmolVLA (2506.01844) all
   report their target-task/target-embodiment data is a small minority of the training
   mixture (2.4%–non-existent for pretraining in SmolVLA's case) — web/VL/other-embodiment
   data dominates and ablations consistently show removing it hurts generalization most.
5. **Human/VR trajectories entered the pretraining and few-shot-adaptation loop.** GR-3
   (2507.15493) formalizes VR-collected human hand trajectories as a first-class,
   faster-to-collect (450 vs 250 traj/hour) data source for few-shot adaptation — distinct
   from egocentric-video "human video" pretraining used elsewhere.
6. **Humanoid whole-body control is being retrofit onto tabletop-manipulation VLAs**
   rather than trained from scratch. HAF (2608.16837) explicitly repurposes π0.5 with a
   hierarchical wrapper instead of building a new humanoid foundation model, a cheaper
   pattern than GR00T N1's from-scratch humanoid pretraining.
7. **In-context/interaction-conditioned adaptation without fine-tuning.** Reflective VLA
   (2606.25215) is the clearest 2026 example: closing distribution-shift gaps via
   observation-action-consequence context at inference time rather than via more
   pretraining data or test-time weight updates.
8. **Efficiency-first, small-model VLAs are now competitive, not just "cheap but worse."**
   SmolVLA (0.45B) and X-VLA (0.9B) both post benchmark numbers matching or beating
   3B+-parameter VLAs on LIBERO/SimplerEnv, reversing the "OpenVLA-beats-RT-2-X-at-7×-fewer-
   params" story from 2024 into a now-normal expectation.

## 6. Limitations / what is still unverified

- **GR00T N1.5 and N1.6**: no arXiv paper exists; architecture/parameter claims for N1.7
  came from WebFetch of the NVIDIA/Isaac-GR00T GitHub repo, not a peer-reviewed source, and
  no quantitative before/after benchmark numbers could be found there — **not verified**.
  Numbers attributed to "GR00T-N1.5"/"GR00T-N1.6" elsewhere in this note (§2, §3, Quick-Facts)
  are third-party baseline-table citations from other papers (FoMoVLA 2607.14739, NebulaVLA
  2608.16503), not numbers taken from an NVIDIA source — flagged as such, and should be
  treated as lower-confidence secondary reports, not primary claims.
- **Figure Helix**: blog-only, no arXiv paper, no independently verifiable benchmark
  numbers. All figures (500 hours training data, "thousands of novel items", 7B/80M
  parameter counts, 200 Hz/7–9 Hz frequencies) are as self-reported by Figure AI's blog post
  and could not be cross-checked against a paper — **not verified** in the sense of
  independent replication, though the text itself was directly fetched and quoted
  accurately.
- **Which specific claims in this note trace only to other papers' reported baselines,
  not the primary source**: GR00T-N1.5 LIBERO avg 86.5% and RoboCasa 47.6%-for-"N1.6" (from
  FoMoVLA, 2607.14739, Table 1/3); GR00T-N1.5 LIBERO-Plus 59.0% (from NebulaVLA, 2608.16503,
  Fig. 3). These are third-party citations, useful as approximate signal but not verified
  against an NVIDIA-authored source.
- No claim in this note was written from memory/training knowledge without a corresponding
  fetched citation; every number above traces to a specific arXiv id and (where quoted)
  table/figure/section fetched via alphaXiv in this session, except the two explicitly
  blog-sourced models (GR00T N1.5/N1.6, Figure Helix), which are flagged inline throughout.

## 7. Sources

- π0-FAST — arXiv:2501.09747 — https://arxiv.org/abs/2501.09747
- π0.5 — arXiv:2504.16054 — https://arxiv.org/abs/2504.16054
- OpenVLA-OFT — arXiv:2502.19645 — https://arxiv.org/abs/2502.19645
- SmolVLA — arXiv:2506.01844 — https://arxiv.org/abs/2506.01844
- Gemini Robotics — arXiv:2503.20020 — https://arxiv.org/abs/2503.20020
- RDT-1B — arXiv:2410.07864 — https://arxiv.org/abs/2410.07864
- GR-3 — arXiv:2507.15493 — https://arxiv.org/abs/2507.15493
- X-VLA — arXiv:2510.10274 — https://arxiv.org/abs/2510.10274
- GR00T N1 (base paper; N1.5/N1.6/N1.7 undocumented in arXiv) — arXiv:2503.14734 —
  https://arxiv.org/abs/2503.14734
- GR00T N1.5/N1.6/N1.7 repo (blog/GitHub, non-peer-reviewed) —
  https://github.com/NVIDIA/Isaac-GR00T
- Figure Helix (blog, non-peer-reviewed) — https://www.figure.ai/news/helix
- Pelican-VLA 0.5 — arXiv:2607.06655 — https://arxiv.org/abs/2607.06655
- FoMoVLA — arXiv:2607.14739 — https://arxiv.org/abs/2607.14739
- HAF — arXiv:2608.16837 — https://arxiv.org/abs/2608.16837
- NebulaVLA — arXiv:2608.16503 — https://arxiv.org/abs/2608.16503
- π0-EqM — arXiv:2605.23128 — https://arxiv.org/abs/2605.23128
- Reflective VLA — arXiv:2606.25215 — https://arxiv.org/abs/2606.25215
