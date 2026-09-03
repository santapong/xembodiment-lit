# Cross-Embodiment Transfer: How It Actually Works, and the Evidence For and Against

## 1. The Problem

Robot learning datasets are individually tiny (100K–1M trajectories at best) compared to vision/language corpora,
so the field's bet — following NLP/CV — is that pooling data across many different robots ("embodiments") produces
positive transfer: a policy trained on the union generalizes better than one trained on any single robot's data
(arXiv:2310.08864). The bottleneck is heterogeneity, not scale: robots differ in (a) action space — end-effector vs.
joint control, position vs. velocity, 2-DoF navigation waypoints vs. 18-DoF bimanual+torso vectors; (b) observation
space — number/placement of cameras, presence of proprioception, image resolution; (c) control frequency — 2 Hz
navigation to 50 Hz dexterous bimanual manipulation; (d) dynamics and embodiment-specific kinematics, where the same
normalized action number can mean opposite physical motions on two robots (arXiv:2310.08864, Sec. IV-A). Every
mechanism reviewed below is fundamentally a way of answering: what gets shared across embodiments, and what must
stay embodiment-specific?

## 2. Mechanism Taxonomy

| Mechanism | Representative papers | What it aligns | Cost |
|---|---|---|---|
| Coarse action/obs unification (fixed 7-DoF EEF vector, one camera) | OXE/RT-X (2310.08864) | Forces all embodiments into one shared vector; ignores true heterogeneity | Cheap, but throws away embodiments that don't fit (quadrupeds, navigation) |
| Per-dataset action normalization (quantile/Gaussian) | OXE, OpenVLA (2406.09246), FAST (2501.09747), Re-Mix (2408.14037) | Removes scale mismatches across embodiments' raw action units | Cheap; does not fix semantic misalignment |
| Modality-specific tokenizers + shared transformer trunk, variable-length I/O | Octo (2405.12213), CrossFormer (2408.11812) | Lets each embodiment keep its native obs/action shape via per-embodiment "readout"/action heads | Moderate; needs new head per new action space |
| Embodiment-specific stems/heads around a shared trunk | HPT (2409.20537), GR00T N1 (2503.14734) | Compresses arbitrary proprioception/vision into fixed-size tokens per embodiment; trunk is universal | Moderate; small per-embodiment adapter, most params shared |
| Data-driven mixture reweighting | Octo weights, Re-Mix (2408.14037) | Which datasets/embodiments get how much gradient signal | Needs a second training run (Re-Mix) or heuristics (Octo) |
| Frequency-domain/compression action tokenization | FAST (2501.09747) | Removes redundancy so one tokenizer scheme works across control frequencies (2–50 Hz) and DoF counts | Cheap; a BPE-style tokenizer, not learned end-to-end |
| Learned universal/latent action space | UniAct (2501.10105), LAPA (2410.11758), GR00T latent actions (2503.14734) | Replaces raw heterogeneous actions with a shared discrete/continuous code, decoded per embodiment | Expensive (extra VQ-VAE/codebook training stage), but action-label-free, so works on human video |
| Continuous flow/diffusion action expert, zero-padded to max dims | π0 (2410.24164), GR00T N1 (2503.14734), ZR-0 (2606.30552) | Shared attention/backbone, embodiment differences absorbed by padding + a small expert | Moderate; padding wastes capacity but is simple |
| Explicit dynamics-prior pretraining before action learning | DyPES-VLA (2608.06374), Action Priors (2606.26095) | Learns "what motion looks like" from action-only or video-only data before cross-modal/cross-embodiment alignment | Two-stage pipeline, extra pretraining stage |
| High-level reasoning as the shared layer, not the action space | Dense ECoT / ZR-0 (2606.30552) | Shares embodiment-agnostic *reasoning* (scene, plan, sub-task) in the VLM; only the action head is embodiment-specific | Needs dense per-frame CoT annotation (~1000s of GPU-annotator-hours) |

## 3. Per-Mechanism Detail

### 3.1 OXE's coarse 7-DoF unification (2310.08864)
RT-X converts each of 22 embodiments' native controllers into a 7-D end-effector vector (x,y,z,roll,pitch,yaw,
gripper), picks one canonical camera, normalizes actions per-dataset before discretizing into 256 bins/dim, and
explicitly does **not** align coordinate frames or units across robots — "the same action vector may induce very
different motions for different robots" (2310.08864, Sec. IV-A). The training mixture for the released RT-1-X/RT-2-X
models used only 9 of the 22 embodiments (RT-1, QT-Opt, Bridge, Task-Agnostic Robot Play, Jaco Play, Cable Routing,
RoboTurk, NYU VINN, Austin VIOLA, Berkeley Autolab UR5, TOTO, Language Table), not the full dataset, because it
predated later additions. Ablation (Table II, 2310.08864): removing the Bridge dataset from the RT-2-X mixture
collapsed "emergent skills" evaluation from 75.8% to 42.8%; going from the 55B RT-2-X to a 5B version dropped
emergent-skills success from 75.8% to 44.4%; removing web-VLM pretraining from a 5B model trained from scratch
dropped emergent-skills success to 0% and generalization to 1%.

### 3.2 Octo: readout heads + heuristic mixture (2405.12213)
Octo processes observations/tasks with modality-specific tokenizers, then inserts learned "readout" tokens that
passively attend to the sequence and feed lightweight per-embodiment action heads; adding a new observation or
action type at fine-tuning time only requires a new small encoder/head, with the pretrained transformer weights kept
frozen (2405.12213, Sec. III-A, Fig. 2). Its 25-dataset training mixture is not simple size-proportional sampling:
after filtering out datasets with no images or non-delta-EEF control, Octo manually buckets datasets into "more
diverse" (double-weighted) vs. "less diverse," and down-weights repetitive datasets — the largest three sources
(Fractal/Kuka/Bridge) still get 17% each (2405.12213, Sec. III-B, Table III). Ablation (Table II): switching from
the RT-X data mix to Octo's own wider mixture raised WidowX success from 60% to 83%; using discretized instead of
continuous+diffusion action prediction dropped success from 83% to 18%.

### 3.3 OpenVLA: quantile normalization, mixture inheritance, and a clean ablation (2406.09246)
OpenVLA discretizes actions per-dimension into 256 bins using the 1st/99th percentile (not min/max) of the training
distribution to be robust to outliers (2406.09246, Sec. 3.2). Its training mixture directly reuses Octo's heuristic
mixture weights for all overlapping datasets, then adds DROID at a conservative 10% weight — and even removes DROID
entirely for the final third of training because "action token accuracy on DROID remained low throughout training,"
suggesting negative interference from a mismatched (higher-frequency) dataset at the chosen weight/model capacity
(2406.09246, Sec. 3.3). The cleanest ablation for cross-embodiment value: OpenVLA trained on the full OXE mixture
scored 76.3% mean success on 8 BridgeData tasks; the same architecture trained only on BridgeData ("OpenVLA-Bridge",
i.e. with OXE cross-embodiment pretraining removed) scored 45.6% — a 30-point absolute drop from removing
cross-embodiment data (2406.09246, Appendix D.1, Table 9).

### 3.4 FAST: tokenization has to change with control frequency (2501.09747)
Per-dimension binning (used by OpenVLA, RT-2, RT-2-X) fails outright above ~20 Hz because consecutive action tokens
become almost perfectly correlated, driving the next-token training signal toward zero — the model collapses to
copying the previous action. FAST instead applies DCT-based frequency compression + BPE to the *normalized* action
chunk. Measured token compression vs. naive binning: BridgeV2 (7-DoF, 5 Hz) 1.75x, DROID (7-DoF, 15 Hz) 3.6x, Table
Bussing (7-DoF, 20 Hz) 5.0x, bimanual T-shirt folding (14-DoF, 50 Hz) 13.2x (2501.09747, Table I). A universal
FAST+ tokenizer trained on 1M cross-embodiment trajectories matches per-dataset tokenizers and, combined with π0,
trains an autoregressive VLA that matches diffusion-π0 performance while using 5x less compute (2501.09747, Sec.
VI-F). Relevant to normalization: FAST normalizes each action dimension by its 1st/99th percentile before DCT
specifically to make tokenization "of cross-embodied datasets with different action scales easier" (Sec. V-B).

### 3.5 GR00T N1: embodiment-specific encoders/decoders + data pyramid (2503.14734)
Each embodiment gets its own MLP state/action encoder and decoder around a shared cross-attention Diffusion
Transformer (System 1) conditioned on a shared VLM (System 2, Eagle-2) — "embodiment-aware state and action encoder"
(2503.14734, Fig. 3, Sec. 2.1). The claimed "+40%" synthetic-data figure in this repo's earlier survey note is
**not verified**: the fetched paper text does not contain a +40% number anywhere. The actual reported synthetic-data
ablation is co-training with generated "neural trajectories" giving **+4.2%, +8.8%, +6.8%** average success-rate
gains on RoboCasa at 30/100/300-demo regimes respectively, and **+5.8%** average on the real GR-1 humanoid
(2503.14734, Sec. 4.4, Fig. 9) — an order of magnitude smaller than the "+40%" previously logged. Separately,
GR00T-N1-2B beats Diffusion Policy by 32.4 points with only 10% of real teleop data and by 30.4 points with the full
dataset (Table 3), and GR00T-N1-2B trained on 10% of real data underperforms Diffusion Policy on full data by only
3.8 points — evidence of strong data efficiency from pretraining, separate from the synthetic-data claim.

### 3.6 CrossFormer: no manual alignment at all (2408.11812)
CrossFormer explicitly refuses to align action/observation spaces across its 20 embodiments (single/bimanual arms,
navigation robots, quadrupeds, quadcopters); instead a transformer ingests any available observation tokens and
predicts through one of four embodiment-specific "action heads" (single-arm 7-D, navigation 2-D waypoint, bimanual
14-D joint, quadruped 12-D joint) (2408.11812, Sec. 3). Averaged over 6 real robots, CrossFormer scores 73% vs. 67%
for the same architecture trained single-embodiment and vs. 51% for the best prior specialist method per robot
(2408.11812, Fig. 5, Table 3) — the paper's own framing is that this shows "no negative transfer," not necessarily
strong positive transfer: "our results do not yet show significant positive transfer across embodiments" (Sec. 5,
Discussion). Against Yang et al.'s manually-aligned nav/manipulation policy, CrossFormer wins 3x on average — evidence
that *not* forcing a common action format can beat forcing one.

### 3.7 HPT: stems/trunk/heads scaling (2409.20537)
HPT tokenizes each embodiment's proprioception and vision into a fixed 16 tokens via a small attention-based "stem,"
shares a transformer "trunk" across all 52 datasets/embodiments, then applies per-task MLP "heads" (2409.20537, Fig.
2–3). Scaling to 52 datasets and 1.1B trunk params improves downstream fine-tuned success by >20% over from-scratch
baselines on multiple sim benchmarks (abstract). Adding simulation and human-video data on top of the 27-dataset
default corpus does produce measurable gains (Fig. 8) despite large embodiment gaps, but the paper is candid that its
core metric — pretraining validation loss — has known caveats and does not cleanly predict closed-loop task success
(Appendix D discussion).

### 3.8 LAPA: label-free latent actions transfer, but not uniformly (2410.11758)
LAPA trains a VQ-VAE to extract discrete "latent actions" from consecutive video frames — no ground-truth robot
actions needed — then pretrains a VLM to predict these tokens, finally fine-tuning a small action head on a handful
of labeled trajectories to map latent→real actions. On OXE pretraining, LAPA beats OpenVLA (also OXE-pretrained) on
2 of 3 real-world manipulation tasks and on all generalization types on average — 50.1% vs. 43.9% (2410.11758, Table
2) — using only 272 H100-hours vs. OpenVLA's 21,500 A100-hours (~30-40x more compute-efficient). But LAPA
*underperforms* OpenVLA specifically on pick-and-place — the task type dominated by grasping precision — because
"most failures of LAPA are due to early grasping" (Sec. 4.4): latent actions capture coarse motion well but are
weaker on fine-grained contact events. LAPA pretrained purely on human video (no robot data at all) still beats
Scratch and even beats OpenVLA(Bridge) on average, evidence that action-label-free cross-embodiment (here,
human→robot) transfer is real but selectively weak.

### 3.9 π0: co-training ablation is implicit, not isolated (2410.24164)
π0's pretraining mixture is 9.1% open-source cross-embodiment data (OXE, Bridge, DROID) plus 903M timesteps of
Physical Intelligence's own 7-platform/68-task data, zero-padded to the largest embodiment's 18-D action/config
vector, with per-task-robot combinations weighted by n^0.43 to down-weight over-represented tasks (2410.24164, Sec.
V-A, Fig. 4). The paper does **not** report a clean "with vs. without cross-embodiment data" ablation for the main
model; the closest evidence is that OpenVLA and Octo, retrained on π0's *same* full cross-embodiment mixture, still
substantially underperform π0 out-of-box (Fig. 7) — showing architecture (flow-matching + action chunking) matters
at least as much as the mixture itself. π0 does show pre-training is "especially useful with harder tasks" via a
scratch-vs-pretrained-then-fine-tuned ablation on complex multi-stage tasks (Fig. 13, Sec. VI-D) — not a
cross-embodiment claim per se, but a pretraining-value claim.

### 3.10 UniAct and latent universal action spaces (2501.10105)
UniAct learns a 256-code VQ codebook of "universal actions" shared across 28 embodiments via a shared VLM, then
decodes to embodiment-specific commands with lightweight per-embodiment MLP heads. A 0.5B UniAct model beats a 7B
OpenVLA on LIBERO by 17.2 points average and beats Octo by 33.6 points (2501.10105, Sec. 4.2); on real WidowX tasks
it wins on visual/motion/physical generalization but *loses* to OpenVLA on semantic generalization and language
grounding (Table 5) — the smaller backbone trades semantic/language capability for action-space efficiency. Fast
adaptation to an unseen robot (AIRBOT, 4 different controller interfaces) needs only 0.8% of total parameters
(4M/500M) fine-tuned, vs. 1.4% for OpenVLA and 2% for Octo.

### 3.11 CrossFormer/HPT/DyPES-VLA: shared attention + routed per-embodiment experts (2408.11812, 2409.20537, 2608.06374)
DyPES-VLA generalizes the stems/heads idea: shared cross-/self-attention layers process a common "query state"
representation, while a Mixture-of-Experts action head routes to embodiment-specific feed-forward experts and
encoder/decoder pairs by a static embodiment-metadata router (2608.06374, Sec. 3.4). Ablating the MoE head down to
one shared dense head costs 1.2 points on RoboTwin-2.0 and 2.1 points on RoboCasa-GR1 (2608.06374, Table 5) —
modest but consistent evidence that some embodiment-specific capacity beats a fully shared action head.

### 3.12 Dynamics-prior pretraining before cross-modal alignment (2608.06374, 2606.26095)
Two 2026 papers converge on the same idea: don't let the action module learn "how to move" and "how to align with
vision/language" simultaneously from scratch. DyPES-VLA pretrains on a future-frame-prediction objective across
action-free human+robot video before action-labeled co-training; removing this future-prediction objective costs 2.4
points on RoboTwin-2.0 and 2.5 points on RoboCasa-GR1 (2608.06374, Table 5, Q2 ablation). Action Priors (2606.26095)
instead pretrains a flow-matching encoder-decoder purely on unconditioned action trajectories (no vision/language at
all) before VLA training; this raises the overall 13-task cross-embodiment average from 55.3% (no prior) to 64.9%,
and adding a compressed-history token pushes it to 68.0% (2606.26095, Table III). The gains are **concentrated on
the data-scarce long tail**: on 4 real-Franka tasks with only 50 demos each (7.6% of the total training mixture),
success rises from 35.0% to 61.3% (prior) to 66.3% (prior+history) — precisely the low-data-per-embodiment regime a
hobbyist would be in.

### 3.13 Dense embodied chain-of-thought as the shared layer (2606.30552)
ZR-0 argues the *transferable* thing across embodiments is not the action space but the high-level reasoning
(scene description, task progress, future plan, atomic sub-task decomposition, target-object grounding) — all
expressed in embodiment-agnostic natural language, with the low-level action head kept separate and
embodiment-specific via zero-padding to 64 dims. Ablating away ECoT pretraining drops LIBERO-10 (the long-horizon
suite) from 96.4% to 92.6% and the LIBERO average from 97.8% to 95.7% (2606.30552, Table 5). Notably ZR-0
*underperforms* π0.5 on a task requiring fine motor precision (Hang Cups: 70.0% vs 85.0%), while beating π0.5 on
tasks requiring more planning/perception — the paper's own read is that "highly precise manipulation may depend more
on the scale of action supervision" than on reasoning-layer alignment (Sec. 4.2, Discussion).

## 4. Ablation-Evidence Table (Positive AND Negative Results)

| Paper | Setup | With cross-embodiment / mechanism | Without / baseline | Delta | Caveat |
|---|---|---|---|---|---|
| OXE/RT-2-X (2310.08864) | Google-Robot "emergent skills" eval | RT-2-X w/ Bridge (WidowX) data in mixture | RT-2-X w/o Bridge | 75.8% → 42.8% (−33 pts) | Removing *one* other-embodiment dataset causes a large drop, but only tested for the 55B model |
| OXE/RT-1-X (2310.08864) | Large in-distribution data domains (Bridge, RT-1) | RT-1-X (co-trained on 9 embodiments) | RT-1 (single embodiment) | RT-1-X *underperforms* RT-1 in this regime | Negative transfer / underfitting when model capacity (35M) is too small for the pooled data diversity |
| OpenVLA (2406.09246) | 8 BridgeData tasks | OpenVLA (full OXE pretrain) | OpenVLA-Bridge (single-embodiment pretrain) | 76.3% → 45.6% (−30 pts) | Clean ablation isolating cross-embodiment pretraining alone |
| Octo (2405.12213) | WidowX eval | Octo's curated 25-dataset mixture | RT-X's 9-dataset (2310.08864-style) mixture | 60% → 83% | Mixture curation matters more than raw dataset count |
| CrossFormer (2408.11812) | 6-robot real average | Joint cross-embodiment training | Single-embodiment training (same architecture) | 67% → 73% | Paper explicitly frames this as "no negative transfer," not strong positive transfer |
| Re-Mix (2408.14037) | RT-X mixture, WidowX+Franka eval | Learned DRO-optimized domain weights | Human-expert-curated (RT-X) weights | 0.42 → 0.80 avg (+38% relative over uniform, +32% over human) | Requires training the full model twice (reference + DRO) to get the weights |
| GR00T N1 (2503.14734) | RoboCasa, 30/100/300-demo regimes | + synthetic "neural trajectory" co-training | Real-robot-only training | +4.2% / +8.8% / +6.8% | This is the *correct* number for the "synthetic data helps" claim — NOT +40% as an earlier internal note stated (unverified/incorrect) |
| LAPA (2410.11758) | Real-world pick-and-place (Open-X pretrain) | LAPA (label-free latent-action pretraining) | OpenVLA (ground-truth-action pretraining) | LAPA *underperforms* on pick-and-place specifically | Latent actions transfer coarse motion well, grasping precision poorly |
| DyPES-VLA (2608.06374) | RoboTwin-2.0 / RoboCasa-GR1 | Embodiment-specific MoE action head | Shared dense action head | +1.2 / +2.1 pts | Small but consistent; supports keeping *some* embodiment-specific capacity |
| Action Priors (2606.26095) | 4 real-Franka long-tail tasks (50 demos each) | Action-prior pretraining + history | No action prior (standard joint VLA training) | 35.0% → 66.3% | Largest verified positive-transfer-adjacent effect in this set, concentrated exactly in the low-data-per-embodiment regime |
| UniAct (2501.10105) | LIBERO (130 tasks) | UniAct-0.5B (universal action space, 28-embodiment pretrain) | OpenVLA-7B (coarse-aligned action space, single robot arm class) | +17.2 pts (0.5B beats 7B) | Confounds architecture size with representation choice — not an isolated ablation |
| Octo (2405.12213) | WidowX, action-representation ablation | Continuous diffusion action head | Discretized action prediction (same data/arch) | 83% → 18% | This is an action-*representation* ablation, not cross-embodiment per se, but it's larger than any mixture-weighting effect found in this survey |

## 5. What We Actually Know

1. **Positive transfer from pooling cross-embodiment data is real but capacity-gated.** OXE/RT-X shows RT-1-X
   (35M params) *underfits* and loses to single-embodiment training on large datasets, while the 55B RT-2-X gains
   from the same pooled data (2310.08864). Small models trained on very heterogeneous mixtures can go backward.
2. **The single cleanest isolated ablation in the literature (OpenVLA-Bridge vs. OpenVLA) shows a 30-point absolute
   success-rate gain from cross-embodiment OXE pretraining** (2406.09246) — this is the best-supported "cross-embodiment
   transfer works" data point in this note, better isolated than most mixture-reweighting claims.
3. **How you unify action spaces matters less than whether you force unification at all.** CrossFormer's
   "no manual alignment, use per-embodiment action heads" beats Yang et al.'s manually-aligned action space by 3x
   (2408.11812) — over-aggressive coarse alignment (as in OXE's original 7-DoF-for-everyone scheme) can destroy signal.
4. **Data mixture weighting is a bigger lever than most architectural choices.** Re-Mix's learned weights beat
   human-expert weights by 32% relative and uniform weights by 38% relative on the exact same architecture and data
   (2408.14037) — bigger effect size than most of the encoder/tokenizer choices reviewed here.
5. **Action tokenization/representation choice can dominate everything else.** Octo's diffusion-vs-discretized
   ablation (83% vs 18%) and FAST's fix for high-frequency binning collapse are larger effects than most
   cross-embodiment mixture or architecture ablations in this survey. Don't spend effort on cross-embodiment mixture
   tuning before getting action representation right.
6. **Negative transfer is under-reported but real and specific, not general.** It shows up as: (a) underfitting at
   small model scale (RT-1-X, 2310.08864); (b) task-specific interference — LAPA's latent actions hurt fine-grained
   grasping specifically while helping coarse motion (2410.11758); (c) OpenVLA's own DROID mixture-weight struggle,
   requiring the dataset to be dropped from training entirely (2406.09246, Sec. 3.3). No paper here reports a broad,
   architecture-independent "cross-embodiment data made things worse across the board."
7. **The value of cross-embodiment pretraining is concentrated in the low-data-per-embodiment regime.** Action
   Priors' biggest verified gain (35.0% → 66.3%) is exactly on 4 real tasks with 50 demos each, 7.6% of the training
   mixture (2606.26095). This is the most directly relevant finding for a hobbyist adding a new low-cost arm.
8. **The "+40% from synthetic data" figure previously in this repo's survey.md is not supported by the source text**
   — the correctly-sourced GR00T N1 ablation gives +4.2/+8.8/+6.8% (sim) and +5.8% (real) for synthetic-data
   co-training (2503.14734, Fig. 9). Treat any bare "+40%" claim about GR00T N1 as an error until re-verified against
   the paper directly.

## 6. Open Questions for a Hobbyist With a Single Custom Low-Cost Arm

- **Does adding OXE data help a new embodiment with <100 demos?** The strongest available evidence is indirect:
  Action Priors shows large gains (35%→66%) on real tasks with only 50 demos each, but *within* a jointly-trained
  cross-embodiment mixture, not as OXE-pretrain-then-finetune-on-a-truly-novel-arm. UniAct's AIRBOT experiment is the
  closest direct test — an unseen robot with 100 demos, fine-tuning only 0.8% of parameters (a new decode head) while
  freezing the learned universal-action codebook — and it works, beating from-scratch baselines. This suggests: yes,
  but the mechanism that transfers is the frozen shared *representation* (universal actions, HPT trunk, or an
  OXE/DROID-pretrained VLA backbone), not the raw pooled data itself; a hobbyist arm should fine-tune a pretrained
  checkpoint (OpenVLA, π0, Octo) with LoRA/head-only adaptation rather than mixing raw hobbyist demos into a
  from-scratch cross-embodiment training run.
- **Which pretrained backbone transfers best to a truly novel action space (not in any training mixture)?** Not
  directly tested by anything in this note. OpenVLA's own paper (per the earlier house note) documents LoRA fine-tuning
  recipes for exactly this. UniAct and HPT both claim easy adaptation via a new lightweight head, but neither is
  tested against OpenVLA-LoRA in a controlled comparison.
- **How many demos are "enough" once cross-embodiment pretraining is used?** No paper here runs a demo-count sweep
  specifically for a genuinely novel embodiment (all sweeps are within-mixture or same-embodiment-family, e.g. LIBERO
  suites). Data Scaling Laws (2410.18647) shows demo count matters far less than environment/object diversity for
  *single-task, single-embodiment* generalization — it's plausible the same "diversity beats demo count" pattern holds
  across embodiments, but that paper doesn't test embodiment diversity at all.
- **Does action-label-free pretraining (LAPA-style) matter for a cheap arm with no teleop rig?** If a hobbyist can
  only get phone/webcam video of themselves doing the task, LAPA's human-video-only result (beats OpenVLA(Bridge) on
  average, 2410.11758) is directly relevant and probably the single most actionable mechanism in this note — but
  expect weaker grasping precision specifically.

## 7. Limitations

- Not verified in this pass: 2602.09722 ("Rethinking VLA Scaling: Alignment, Mixture, and Regularization"),
  2603.06450 ("Data Analogies Enable Efficient Cross-Embodiment Transfer"), 2608.18433 ("The Embodiment Gap in Robot
  Foundation Models"), and 2508.06426 ("Shortcut Learning in Generalist Robot Policies") were surfaced by
  `discover_papers` (abstract snippets only) as directly on-topic negative/critical results but were **not** fetched
  with `answer_pdf_queries` due to the two-call `discover_papers` budget and per-paper fetch budget in this pass —
  their numbers are not cited here and should be pulled in a follow-up note.
  Prompt-specified paper "any paper with explicit negative-transfer findings in OXE mixtures" search surfaced these
  four candidates but none was deep-fetched; the closest verified negative-transfer evidence instead comes from
  RT-1-X underfitting (2310.08864) and LAPA's grasping-specific weakness (2410.11758).
- `discover_papers` for "UniAct" initially returned an unrelated 2025-12 humanoid motion-generation paper; the correct
  UniAct (universal-action-space) paper was found at 2501.10105 via a direct title lookup, not via `discover_papers`.
- The GR00T N1 "+40%" figure could not be found anywhere in the fetched paper text (report or full text); this note
  treats the number as an error in this repo's prior survey.md and reports the verified figures instead.
- π0's paper does not contain an isolated "with vs. without cross-embodiment data" ablation for its main
  700k-step model; the comparison offered here (π0 vs. OpenVLA/Octo retrained on the same mixture) is the closest
  available evidence but conflates architecture with data.
- CrossFormer's "up to 1400 action dimensions" and 20-vs-30-embodiment count discrepancy between its abstract
  banner and body text (Sec. 3.1 says 20 embodiments) is reported as stated in the source; not independently resolved.

## 8. Sources

All numeric claims above are tagged inline with arXiv ids and were fetched via `answer_pdf_queries` /
`get_paper_content` (alphaXiv MCP), except where explicitly marked "not verified."

- 2310.08864 — Open X-Embodiment: Robotic Learning Datasets and RT-X Models
- 2405.12213 — Octo: An Open-Source Generalist Robot Policy
- 2406.09246 — OpenVLA: An Open-Source Vision-Language-Action Model
- 2501.09747 — FAST: Efficient Action Tokenization for Vision-Language-Action Models
- 2503.14734 — GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- 2408.11812 — Scaling Cross-Embodied Learning (CrossFormer)
- 2409.20537 — Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers (HPT)
- 2501.10105 — Universal Actions for Enhanced Embodied Foundation Models (UniAct)
- 2410.11758 — Latent Action Pretraining from Videos (LAPA)
- 2410.24164 — π0: A Vision-Language-Action Flow Model for General Robot Control
- 2408.14037 — Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning
- 2410.18647 — Data Scaling Laws in Imitation Learning for Robotic Manipulation
- 2608.06374 — DyPES-VLA: Learning Shared Dynamics Priors and Embodiment-Specific Control for Cross-Embodiment Manipulation
- 2606.26095 — Learning Action Priors for Cross-embodiment Robot Manipulation
- 2606.30552 — Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision (ZR-0)
- Surfaced but not deep-fetched (abstract only, via `discover_papers`): 2602.09722, 2603.06450, 2608.18433, 2508.06426
