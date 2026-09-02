# X-Embodiment Literature Workspace

Research notes and verified references on embodied AI with the **Open X-Embodiment dataset** (arXiv:2310.08864) and the vision-language-action (VLA) model lineage it enabled.

## Contents

- `paper/references.bib` — 101 BibTeX entries, all fetched from arXiv metadata (the 10 anchors, every paper cited in the deep dives, and the 2026-09-03 transcript ingest)
- `notes/survey.md` — field survey: OXE foundation → open generalist policies → newer wave (2024–2026), with key numbers and themes
- `notes/openvla-deep-dive.md` — OpenVLA architecture, training data, results, efficiency methods, limitations
- `notes/frontier-2025-26.md` — what came after the anchors: π0-FAST, π0.5, OpenVLA-OFT, SmolVLA, Gemini Robotics, RDT-1B, GR-3, X-VLA, GR00T N1.x, Helix, mid-2026 wave
- `notes/cross-embodiment-transfer.md` — how cross-embodiment transfer is engineered, mechanism taxonomy, ablation evidence for and against
- `notes/evaluation-and-failure.md` — benchmarks, statistical rigor, runtime failure detection, reactivity vs chunking, safety evals, rigorous-eval checklist
- `notes/finetune-own-arm.md` — practical recipe + budget for fine-tuning an open VLA on one custom low-cost arm
- `notes/world-models.md` — world models as planners and as data engines; three-axis taxonomy; the verified Xiaomi-U0 augmentation result and its caveats
- `notes/humanoid-whole-body.md` — humanoid loco-manipulation: control stack, model-based vs learning scorecard, hardware limits, and why VLA is not yet the answer there

**Reading order:** `survey.md` → `openvla-deep-dive.md` → `frontier-2025-26.md` → `cross-embodiment-transfer.md` → `evaluation-and-failure.md` → `finetune-own-arm.md`. Then `world-models.md` and `humanoid-whole-body.md`, which are newer and thinner: each has one paper read in full and the rest filed as an explicitly unverified reading map.

Every number in the notes carries the arXiv id it was read from; claims that could not be verified against a fetched source are marked "not verified".

## Quick Facts

| Model | Params | Data | Headline result |
|---|---|---|---|
| RT-2-X | 55B | OXE mixture | ~3× generalization vs single-embodiment |
| Octo | 27M–93M | 800k OXE trajs | First fully open generalist policy |
| OpenVLA | 7B | 970k OXE trajs | Beats closed RT-2-X by 16.5% absolute |
| π0 | ~3B VLM + action expert | 7 platforms | Flow matching @ 50Hz, dexterous SOTA |
| GR00T N1 | 2.2B | real+video+synthetic | Open humanoid foundation model |

### Post-2024 (from `notes/frontier-2025-26.md`)

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

## Key Resources

- Dataset: https://robotics-transformer-x.github.io
- Code: https://github.com/google-deepmind/open_x_embodiment
