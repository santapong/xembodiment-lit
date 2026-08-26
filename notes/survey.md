# X-Embodiment & Embodied AI — Research Survey
Updated: 2026-08-26

## 1. Foundation: Open X-Embodiment (arXiv:2310.08864, ICRA 2024)
- Community dataset from 21 institutions: **1M+ trajectories, 22 embodiments, 60 source datasets, 34 labs**, standardized in RLDS (TFRecord) format. 527 skills / 160,266 tasks.
- **RT-1-X**: RT-1 Transformer on a 9-manipulator mixture → +50% success vs original per-lab SOTA policies.
- **RT-2-X**: RT-2 VLM (PaLI-X backbone, 55B) co-fine-tuned with web data → ~3× generalization over single-embodiment training; emergent semantic skills.
- Central claim: cross-embodiment positive transfer is real; robotics can follow the NLP/CV "pretrained backbone" consolidation path.
- Resources: robotics-transformer-x.github.io | github.com/google-deepmind/open_x_embodiment

## 2. Open Generalist Policies Built on OXE
| Model | arXiv | Params | Data | Headline |
|---|---|---|---|---|
| Octo (RSS'24) | 2405.12213 | 27M/93M | 800k OXE trajs, 25 datasets | First fully open generalist policy; transformer+diffusion head; beats RT-1-X, matches RT-2-X; language + goal-image conditioning; finetunes to new obs/action spaces |
| OpenVLA (CoRL'24) | 2406.09246 | 7B | 970k OXE trajs | Llama 2 + DINOv2/SigLIP; beats closed RT-2-X by 16.5% absolute across 29 tasks with 7× fewer params; LoRA fine-tuning on consumer GPUs; beats Diffusion Policy by 20.4% |

## 3. Newer Wave (2024–2026)
- **DROID** (2403.12945, RSS'24): 76k demos / 350h, 564 scenes, 86 tasks, collected by 50 operators worldwide on Franka. Higher quality/diversity than OXE average; now standard pretraining supplement.
- **π0** (Physical Intelligence, 2410.24164): PaliGemma-3B VLM + flow-matching action expert → smooth 50Hz continuous action chunks. Trained across 7 robot platforms / 68 tasks. State of the art for dexterous manipulation. Variant π0-FAST uses frequency-space action tokenization for faster training.
- **GR00T N1** (NVIDIA, 2503.14734): open humanoid foundation model. Dual-system: VLM reasoning (System 2, 10Hz) + diffusion transformer flow-matching actions (System 1, 120Hz). Trained on data pyramid of real robot + human video + synthetic data (co-training with neural trajectories gave **+4.2/+8.8/+6.8 pts** in sim at 30/100/300 demos and **+5.8 pts** on the real GR-1 — 2503.14734 §4.4; an earlier draft of this note said +40%, which the paper does not support). Deployed on Fourier GR-1. Later GR00T N1.x extends to bimanual/semi-humanoid.
- **CogACT** (MSRA/Tsinghua, 2411.19650): componentized VLA — VLM cognition module + separate diffusion action transformer. Beats OpenVLA by >35% (sim) / 55% (real), beats 55B RT-2-X by 18% absolute in sim.
- Field trend (per VLA surveys): lineage RT-1 → RT-2 → RT-2-X → OpenVLA → π0 → CogACT → SpatialVLA/X-VLA/Gemini Robotics/Helix. Pretraining = OXE + DROID (+ RoboMIND, human video). Evaluation migrated to LIBERO/CALVIN/SimplerEnv + real-world AutoEval/RoboArena.

## 4. Themes Across the Field
1. **Flow matching / diffusion action heads replaced discrete tokenization** as the dominant action representation (π0, GR00T, CogACT).
2. **Open beats closed at smaller scale**: OpenVLA 7B > RT-2-X 55B — diversity of data and good components beat raw parameter count.
3. **Data heterogeneity is the bottleneck**: embodiment-specific encoders/decoders (GR00T) and normalization schemes are how the field copes with OXE's heterogeneity.
4. **Efficient adaptation is standard**: LoRA + quantization (OpenVLA) makes VLAs practical on consumer GPUs.

## 5. Relevance to Santapong's interests
- Language-guided robotics (ROS 2/MCP): OpenVLA and Octo checkpoints are runnable and finetunable; LeRobot (HF) ships π0 implementations.
- RLDS format is the interchange standard; OXE colab notebooks allow browsing all datasets without full download (~1.2TB processed for Octo pretraining).

## See also (deep dives added 2026-08-26)
- `frontier-2025-26.md` — everything after the 10 anchor papers: π0-FAST/π0.5, OpenVLA-OFT, SmolVLA, Gemini Robotics, RDT-1B, GR-3, X-VLA, GR00T N1.x, Helix, and the mid-2026 wave.
- `cross-embodiment-transfer.md` — how transfer is actually engineered (action alignment, embodiment adapters, latent actions, data mixtures) and the evidence for *and against* it.
- `evaluation-and-failure.md` — benchmarks, statistical rigor, runtime failure detection, reactivity vs chunking, safety evals; the rigorous-eval checklist.
- `finetune-own-arm.md` — practical recipe and budget for fine-tuning an open VLA on a single low-cost arm.
