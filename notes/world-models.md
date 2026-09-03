# World Models for Embodied AI
Updated: 2026-09-03. Sourcing: two papers fetched in full via alphaXiv (2510.16732, 2607.11643); every number below carries its arXiv id. Papers listed in §4 are **title/abstract only, not verified** — they arrived via an alphaXiv Assistant transcript, and their ids were confirmed to resolve on arXiv but their contents were not read.

## 1. What a World Model Is Here

An internal simulator that captures environment dynamics well enough to support forward and
counterfactual rollouts (2510.16732, §I). The operational definition is a
**compression → prediction → control** pipeline: compress observations into a latent state,
predict how that state evolves under actions, then plan by reasoning over imagined rollouts.

The survey draws a boundary worth keeping: world models yield **actionable** predictions for an
embodied agent, which distinguishes them from static scene descriptors and from purely generative
visual models that do not capture *controllable* dynamics (2510.16732, §I). A video generator that
produces beautiful robot footage is not a world model unless actions condition it.

Formally a POMDP with a learned latent `z_t`, trained on a reconstruction term plus a KL that
aligns the filtered posterior `q(z_t | z_{t-1}, a_{t-1}, o_t)` with the dynamics prior
`p(z_t | z_{t-1}, a_{t-1})` (2510.16732, §II-B, Eq. 1–5). Instantiable with recurrent, Transformer,
or diffusion backbones.

## 2. The Three-Axis Taxonomy (2510.16732, §III)

| Axis | Options | The trade-off it encodes |
|---|---|---|
| **Functionality** | Decision-Coupled vs General-Purpose | Coupled models are task-specific: better sample efficiency and closed-loop performance, weaker generalization off-distribution. General-purpose simulators transfer broadly, but their pretraining objective misaligns with control and is "brittle unless explicitly mitigated" |
| **Temporal modeling** | Sequential Simulation & Inference vs Global Difference Prediction | Sequential is autoregressive: fine-grained control, natural closed-loop, but error accumulates and cost scales linearly with horizon. Global predicts future states in parallel: lower wall-clock latency for long horizons, but weakened closed-loop interactivity and smoothed local dynamics |
| **Spatial representation** | Global Latent Vector, Token Feature Sequence, Spatial Latent Grid, Decomposed Rendering Representation | Sits on a coherence–fidelity–efficiency curve. Latent vector is cheap and real-time but loses spatial detail; token sequences are expressive but need large data/models and high inference cost; BEV/voxel grids preserve local topology at heavy memory cost; 3DGS/NeRF give geometry-consistent, object-controllable views but train expensively and handle rapid dynamics and topology changes badly |

The taxonomy is applied uniformly across robotics, autonomous driving, and general video, which is
what makes it usable as a filing system rather than a reading order.

## 3. World Models as Data Engines — the verified case (2607.11643)

Xiaomi-Robotics-U0 is the concrete instance of the 2026 shift from "world model as planner" to
"world model as scalable data engine."

- **38-billion-parameter** multimodal autoregressive model, jointly optimizing text-to-image,
  image editing, embodied scene generation, embodied transfer, and embodied video generation
  (2607.11643, Abstract).
- Augmenting demonstrations with its generated data moved a π0.5 policy from **36.9% to 63.2%**
  on out-of-distribution real-world manipulation (2607.11643, Abstract and §3.3).

**Read that number carefully — three qualifications the abstract does not carry.**
1. The metric is **task completion progress**, partial credit over ordered milestones, *not* binary
   success (2607.11643, §3.3, "Metric"). A policy that reliably solves early subgoals and fails late
   scores well.
2. 36.9 → 63.2 is the **interference group** only: held-out backgrounds and lighting. In the base
   (in-distribution) group the augmented policy is **81.0 vs 82.1**, i.e. very slightly *worse*, which
   the authors attribute to sharing training capacity with augmentation that adds no new
   in-distribution information (2607.11643, Fig. 17).
3. N is small: three real tabletop tasks, **three layouts × three trials** per task and group
   (2607.11643, §3.3). By the standards in `evaluation-and-failure.md` §3 this is an order of
   magnitude under-powered for a confident effect size, even though the direction is plausible.

The honest summary: **style-transferred augmentation buys visual invariance to novel backgrounds
and lighting, and costs a little in-distribution capacity.** That is a narrower and more useful claim
than "synthetic data improves robot policies."

## 4. Reading map — not verified

Filed in the alphaXiv folder *World models*. Ids confirmed to resolve; contents not read.

| Paper | arXiv | Why it is in the pile |
|---|---|---|
| A Comprehensive Survey on World Models for Embodied AI | 2510.16732 | **fetched** — the taxonomy above |
| World Model for Robot Learning: A Comprehensive Survey | 2605.00080 | second survey; read for disagreements with the three-axis framing |
| Xiaomi-Robotics-U0 | 2607.11643 | **fetched** — the data-engine case |
| MotionWAM | 2606.09215 | world-action modelling toward real-time humanoid loco-manipulation |
| DreamMimic | 2608.22278 | imagined dynamics for contact-rich whole-body behaviour |
| LUCID | 2608.07746 | latent skills + imagined dynamics for long-horizon sequencing |
| ω-0 | 2608.06375 | latent predictive world-action model, concurrent humanoid loco-manipulation |

## 5. Open Challenges the Survey Names (2510.16732, §VI)

- **No unified dataset**, and metrics that reward pixel fidelity over **physical consistency**. Highly
  realistic generative models "capture correlations in appearance without robust causal structure,
  leading to visually plausible yet physically incorrect predictions."
- **Performance vs the compute budget real-time control actually has.**
- **Long-horizon temporal consistency** without error accumulation, the core modelling difficulty.
- Expected direction is *families* of cross-domain multimodal suites assessing prediction, decision
  making and safety under sim-to-real shift, rather than one benchmark.

## Limitations

- Only 2 of the 7 filed papers were read. Everything in §4 beyond the title is unverified.
- The survey's own quantitative comparison (§V) was not fetched, so no cross-model numbers are
  reproduced here.
- No claim is made about whether world-model augmentation helps on *our* hardware; the one verified
  result uses a 38B generator, which is far outside the rented-GPU budget in `finetune-own-arm.md`.

## Practical Notes

- The pixel-fidelity-over-physics critique is the same failure this workspace already documents for
  policy evaluation: an easy-to-measure proxy displacing the thing that matters. See
  `evaluation-and-failure.md` §1.
- For our purposes the useful half of Xiaomi U0 is not the model but the **evaluation split**:
  reporting base and interference groups separately is exactly the discipline that makes an
  augmentation claim checkable. Copy the split, not the 38B model.
- Decision-Coupled + Sequential + Global Latent Vector is the cheap corner of the taxonomy
  (DreamerV3-style) and the only one plausibly trainable on a single rented GPU.

## See also
- `evaluation-and-failure.md` — why the small-N caveat above matters, and Foresight (2606.23085),
  which uses world-model latents for failure detection.
- `humanoid-whole-body.md` — the humanoid loco-manipulation papers that overlap this pile.
- `frontier-2025-26.md` — where world models sit in the wider VLA lineage.
