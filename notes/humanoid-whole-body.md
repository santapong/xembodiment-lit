# Humanoid Whole-Body Control, Learning, and Mechanism
Updated: 2026-09-03. Sourcing: one paper fetched in full via alphaXiv (2501.02116); every number below carries its arXiv id. Papers listed in §5 and §6 are **title/abstract only, not verified** — they came from an alphaXiv Assistant transcript, and their ids were confirmed to resolve on arXiv but their contents were not read.

Opened because the transcripts pushed hard on humanoids and this workspace had **zero** humanoid
coverage. It also connects to the FFW humanoid mirror work outside this repo.

## 1. Why a Humanoid Is Not Just a Bigger Arm

A humanoid is a **high-dimensional, floating-base, hybrid dynamical system**. Three consequences the
manipulation literature never has to face:

- **The base is free.** Nothing is bolted down, so joint torques and environmental contact forces
  jointly produce motion. Everything routes through the floating-base dynamics
  `M(q)q̈ + h(q,q̇) + g(q) = Sᵀτ + J_c(q)ᵀλ`.
- **Contact modes switch.** Double support, left support, right support, flight, hand contact.
  Walking is a hybrid system where the continuous dynamics change at touchdown and liftoff.
- **Feasibility is not kinematics.** A reachable pose can still fall over. Contact forces must stay
  inside the friction cone and the centre of pressure inside the support region.

The anchor survey (2501.02116, Georgia Tech / USC / TUM / DeepMind / Stanford / CMU / NVIDIA and
others, 30 years of literature) is explicitly about **loco-manipulation**, bipedal whole-body motion
rather than multi-finger dexterity.

## 2. The Control Stack (2501.02116)

The field has converged on a **predictive–reactive hierarchy**: a whole-body or centroidal-dynamics
MPC on top, coupled to a local task-space whole-body controller (2501.02116, §I, citing [9]). Both
are posed as optimal control problems and solved numerically; open work is on efficiency, numerical
stability and scaling to high-DoF systems, not on the formulation.

Survey structure, which doubles as the reading order (2501.02116, Fig. 2): tactile sensing (§III) →
contact planning (§IV) → motion planning / MPC (§V) → whole-body control (§VI) → skill learning,
RL and IL (§VII) → foundation models (§VIII).

## 3. Model-Based vs Learning-Based — the survey's own scorecard

Table VI (2501.02116, §IX-A) rates both paradigms. Reproduced because it is unusually blunt:

| | MPC / WBC | RL | IL | VLA |
|---|---|---|---|---|
| Multi-modal sensor flexibility | low | medium | high | very high |
| Robustness | high | very high | low | very low |
| Motion accuracy | very high | medium | medium | very low |
| Real-time feasibility | high | high | high | **low** |
| Generalizability | medium | low | low | high |

Achieved humanoid skills, same table: locomotion is **very high** for both MPC and RL and only
**medium** for IL, with **NA for VLA**. Whole-body multi-contact is **NA** for both IL and VLA.

**The takeaway that matters for anyone planning to point a VLA at a humanoid:** the survey states
that VLA "holds promise for enhanced generalizability; however, its effectiveness in mastering
humanoid skills has yet to be convincingly demonstrated" (2501.02116, §IX-A). It also warns the
comparison itself is soft — "many algorithm performances reported are anecdotal and highly depend
on their implementation," and a concrete numerical comparison is hard for lack of a benchmark
(2501.02116, §IX-A, pointing at §VII-G). **The humanoid field has the same evaluation problem this
workspace documents for VLAs**, and knows it.

Two framing points the survey insists on (2501.02116, §I):
1. Sim-to-real RL relying on an accurate simulator dynamics model **is** arguably model-based, even
   though the RL algorithm does not model dynamics explicitly.
2. Model-based control and sim-to-real RL "do not conflict; on the contrary, they often complement
   each other and can be combined to achieve a better performance than either method alone."

## 4. Hardware Is Load-Bearing

- **Efficiency gap with humans is large.** Cost of transport, energy per unit distance normalized by
  body weight: today's humanoids sit at **COT > 0.7** against **COT = 0.2** for humans
  (2501.02116, §II-A, citing [43]). Early passive dynamic walkers beat humans on efficiency but had
  almost no versatility. This gap is what motivates passive-compliant energy storage and controllers
  that exploit natural dynamics.
- **The torque path is a thermal problem.** High torque needs high current, which strains power
  electronics and overheats; quasi-direct-drive actuators (under 10:1 reduction) buy
  backdrivability and force-control bandwidth at the cost of needing that current
  (2501.02116, §IX-B).
- Missing piece named explicitly: **no scaling law for humanoid foundation models**, the equivalent
  of the LLM compute/data/parameter guidance (2501.02116, §VIII).

## 5. Reading map — control and learning, not verified

Filed in the alphaXiv folder *Humanoid whole-body control*. Ids resolve; contents not read.

| Paper | arXiv | Pile |
|---|---|---|
| Humanoid Locomotion and Manipulation survey | 2501.02116 | **fetched** — everything above |
| A Survey of Behavior Foundation Models for humanoid whole-body control | 2506.20487 | the second anchor survey; read next |
| Expressive Whole-Body Control | 2402.16796 | coordinated full-body motion |
| OpenHLM | 2606.22174 | empirical recipe for whole-body loco-manipulation |
| HAF | 2608.16837 | adapting generalist VLAs to humanoid loco-manipulation |
| FRoM-W1 | 2601.12799 | language-conditioned whole-body control |
| Humanoid Policy ~ Human Policy | 2503.13441 | human behaviour priors → humanoid policies |
| OmniContact | 2606.26201 | chaining meta-skills via contact flow |
| FetchMan | 2608.17027 | visual loco-manipulation from simulated experience |
| Humanoid-DART | 2606.26855 | diffusion-guided augmentation of scarce demos |
| Closing the Loop in Humanoid VLA | 2607.18016 | persistent 3D object tokens |
| Humanoid Occupancy | 2507.20217 | multimodal occupancy perception |
| Advances and opportunities for legged robots | 2607.28952 | wider legged context |
| Legged robotics in non-inertial environments | 2604.20990 | moving/dynamic support surfaces |
| RL for an inline-skating humanoid | 2606.31807 | underactuated contact control |
| frax | 2604.04310 | fast differentiable kinematics/dynamics in JAX |

## 6. Reading map — mechanism and design, not verified

Kept in the same folder rather than split out; the division belongs in the reading order, not the
taxonomy.

| Paper | arXiv | Pile |
|---|---|---|
| Embracing Evolution: body–control co-design | 2510.03081 | why morphology and control should be optimized together |
| Human-Level Actuation for Humanoids | 2511.06796 | torque, speed, power envelopes |
| A Framework for Optimal Ankle Design | 2509.16469 | one real mechanism in detail; parallel ankle trade-offs |
| LEGO | 2604.08636 | kinematic/geometry design optimization |
| Optimal Design of a Walking Robot | 2505.00923 | multi-criteria structural synthesis |
| ergoCub | 2605.26991 | whole-system design for human interaction |
| Towards Robotic Dexterous Hand Intelligence | 2605.13925 | hands, if the target is manipulation not walking |

## Limitations

- **1 of 23 filed papers was read.** Sections 5 and 6 are a filing system, not a synthesis. Do not
  cite anything from them without fetching it first.
- The survey is v2, April 2025, so its "state of the art" predates the 2026 papers listed above; its
  VLA-for-humanoids verdict in §3 should be re-checked against HAF (2608.16837) and
  FRoM-W1 (2601.12799) before being treated as current.
- No claim here about any specific humanoid platform's numbers.

## Practical Notes

- **Read the control stack before the learning papers.** The survey's own advice, and it matches
  the failure mode of starting end-to-end: with a learned policy on a floating-base robot you
  cannot tell whether a failure came from the policy, the dynamics model, the contact model, the
  actuator, calibration, or the mechanism.
- The COT figure is the cleanest single argument that humanoid hardware is unsolved, not just
  humanoid software.
- **The VLA row in §3's table is the honest reason to keep humanoid ambitions separate from the VLA
  eval thread.** Real-time feasibility low, motion accuracy very low, whole-body multi-contact NA.
  Treat humanoid loco-manipulation as a different problem with a different literature, not as VLA
  with more joints.

## See also
- `world-models.md` — MotionWAM, DreamMimic, LUCID and ω-0 sit in both piles.
- `evaluation-and-failure.md` — the benchmark gap named in §3 is the same one documented there.
- `survey.md` — where the humanoid surveys sit relative to the VLA survey layer.
