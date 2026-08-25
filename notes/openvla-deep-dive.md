# Deep Dive: OpenVLA (arXiv:2406.09246, CoRL 2024)

## Motivation
Existing VLAs were closed (RT-2) or lacked efficient adaptation paths. Goal: an open-source generalist VLA like the open-LM ecosystem (LLaMA-style).

## Architecture
- Backbone: **Llama 2 7B** language model.
- Vision: fused features from **DINOv2 + SigLIP** encoders (spatial structure + language-aligned semantics), projected into the LM token space.
- Actions: discretized into tokens via per-dimension bucketing, decoded by the LM head (single next-token prediction per step).
- Total: 7B parameters (vs 55B RT-2-X).

## Training
- Pretrained on **970k real-world demonstrations from Open X-Embodiment** (mixture spanning many embodiments/tasks/scenes).

## Key Results
- Beats **RT-2-X by 16.5% absolute success rate** across 29 evaluation tasks on WidowX + Google Robot — with 7× fewer parameters.
- Beats Diffusion Policy (from-scratch imitation) by **20.4%**, especially on multi-object multi-task settings requiring language grounding.
- Strong generalization: unseen objects, backgrounds, environments.

## Efficiency Contributions
- **LoRA fine-tuning**: adapt to new robots/tasks on consumer GPUs (e.g., single 48GB GPU) without performance loss.
- **Quantization** (4-bit) for serving without downstream success-rate degradation.
- This made VLAs practical outside big labs — arguably OpenVLA's biggest impact.

## Limitations (noted by authors/community)
- Discrete action tokenization → lower control frequency than flow-matching successors (π0, GR00T N1).
- Single-frame observations, no history.
- Successors (CogACT etc.) show dedicated diffusion action heads substantially outperform LM-token decoding at similar scale.

## Practical Notes
- Weights/code/data public (openvla/openvla on GitHub; HF hub).
- Fine-tune recipes included for new embodiments — relevant entry point for personal robotics experiments.
