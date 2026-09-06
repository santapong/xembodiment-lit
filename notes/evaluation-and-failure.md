# Evaluating VLAs Rigorously, and Detecting/Recovering from Failure
Updated: 2026-08-26. Sourcing: alphaXiv PDF fetches only; every number is tagged with its arXiv id from text actually retrieved. Anything not fetched is marked "not verified."

## 1. Why Eval Is the Weak Link

Robot policy evaluation has not scaled with policy capability. The field's default protocol — binary success/failure at a fixed timeout, **N≤25 rollouts per condition**, no confidence intervals, no paired statistical test — is documented directly: a 13-paper survey of 2023–2025 real-robot VLA papers (RT-1, RT-2, RT-X/Open X-Embodiment, OpenVLA, π0, π0.5, π0.6, GR00T N1, ACT/ALOHA, Mobile ALOHA, DROID, SmolVLA, ECoT) finds modal per-condition N of **10–20**, and **none of the 13 report confidence intervals or paired tests** (2605.29710, PhAIL). The one recent counter-example is the "LBM examination," which runs blinded same-session A/B at N=50 real/200 sim per condition with Bayesian posteriors and paired Barnard/Welch tests with Bonferroni correction (cited in 2605.29710 and 2603.13616).

Three compounding problems:
- **Sample sizes are an order of magnitude below what binary tests need.** A Wilson 95% CI on a single observed 70% success rate is [0.40, 0.89] at N=10 and [0.48, 0.86] at N=20 — too wide to defend a ranking claim (2605.29710, Appendix H).
- **Simulated evaluation is a leaky proxy.** SimplerEnv shows validation-set action MSE (the traditional ML model-selection metric) is *negatively correlated* with real-world success (2405.05941, Table I); simulated success rate correlates well only if visual and control disparities are explicitly closed (green-screening + texture matching + system ID), and even then sim-to-real gaps remain policy-dependent (AutoEval finds Open-π0 performs very poorly in SIMPLER on "put eggplant to sink" but does well in the real world — 2503.24278).
- **Success rate is a single scalar that hides distributional shape.** Time-to-success CDFs can cross between two policies, so different "reasonable" scalar summaries (success at threshold τ, UPH, RMST, AUC-vs-human) give opposite top-1 rankings on the *same underlying data* (2605.29710, PhAIL, §5.1: four scalar choices → three different leaders).

## 2. Benchmark Table

| bench | sim/real | tasks | what it measures | known problems | arXiv |
|---|---|---|---|---|---|
| LIBERO | sim (Robosuite/MuJoCo) | 130 tasks, 4 suites (Spatial/Object/Goal/10 [orig. named 90+Long]) of 10–100 tasks each | lifelong-learning knowledge transfer across spatial/object/goal distribution shift; per-task 10-rollout success rate | Designed for *lifelong learning ablations*, not leaderboard comparison; per-task SR reported over just 10 trials/task in the original paper (Table E.3); success on "put X on top of cabinet" swings 0.0–0.85 across seeds/architectures with no CI | 2306.03310 |
| SimplerEnv | sim (SAPIEN, real-to-sim) | Google Robot (pick coke can, move near, open/close drawer, drawer+apple) + WidowX/BridgeData (spoon/carrot/blocks/eggplant), ~7 task families | real-to-sim correlation of manipulation policy success/behavior modes | Only rigid-object tasks (no deformables/liquids at release); requires "Visual Matching" asset tuning per robot to get correlation (raw sim MMRV 0.087 vs 0.031 with tuning); doesn't cover cloth/liquid tasks at all | 2405.05941 |
| CALVIN | sim (PyBullet) | 34 subtasks, chained into 1000 5-step long-horizon sequences over 4 environments | zero-shot language-conditioned long-horizon control | Best 2022 baseline (MCIL) solved chains-of-5 in only **0.08%** of trials, showing the benchmark saturates on short-horizon (53.9%) but is essentially unsolved long-horizon — makes it hard to distinguish "good" policies at the low end | 2112.03227 |
| RoboCasa | sim (MuJoCo/RoboSuite, AI-generated assets) | 100 tasks (25 atomic + 75 LLM-generated composite) across 120 kitchen scenes | scaling trend of imitation learning with human vs. machine-generated (MimicGen) data | Composite-task fine-tuning success is low (`generated trajectories … exhibited undesirable effects, such as jerky motions and collisions`); real-world transfer study uses only 3 tasks, 5 demos/category | 2406.02523 |
| SimplerEnv/LIBERO successors (LIBERO-Plus, LIBERO-PRO, LIBERO-VIFO, LIBERO-Safety) | sim | perturbation/robustness/safety variants of LIBERO | robustness to camera/light/layout/language shift; memorization; safety | See §6 and §8 below — cited as OOD/robustness stress-tests, not raw capability leaderboards | (see refs below) |
| RoboArena | real (DROID platform, distributed) | open-ended (evaluators choose task+scene); 4284 total rollouts across 7 policies, 7 institutions in the released study | crowd-sourced pairwise (A/B, double-blind) real-world policy ranking, Chatbot-Arena-style | Requires physical DROID hardware at each site; convergence needs ~100 pairwise comparisons per policy pair to match an "oracle" ranking; Goodhart's-law risk flagged explicitly as future work | 2506.18123 |
| AutoEval | real (autonomous cells, BridgeData V2 / WidowX) | 5 tasks (drawer open/close, eggplant→sink/basket, cloth fold) | fully autonomous real-world eval (learned reset policy + learned VLM success classifier), no human in the loop | Binary success only (no partial-credit); environment-creation still needs a few hours of human setup per new scene; motor overheating requires 20-min cooldown every 6h | 2503.24278 |

**What "success rate" hides — π0/OpenVLA-OFT on LIBERO (2502.19645):** OpenVLA-OFT reports LIBERO success rates *averaged over 500 trials per task suite (10 tasks × 50 episodes)* — a real improvement in rigor over the LIBERO-original 10 trials/task. Headline numbers: OpenVLA-OFT 97.1% avg (Spatial 97.6 / Object 98.4 / Goal 97.9 / Long 94.5) vs. π0 94.2% vs. π0+FAST 85.5% vs. base OpenVLA 76.5% (2502.19645, Table I). But: (a) results are pooled across a *modified, filtered* training dataset in most rows (near-zero-magnitude actions and unsuccessful demos removed) — a different row uses the *unfiltered* dataset and only reaches 94.5% avg, i.e. the reported SOTA number depends on a training-set curation choice, not just the fine-tuning recipe; (b) no confidence intervals or seed variance are reported at all — 500 trials per suite is enough to *shrink* the Wilson interval to roughly ±2pp at 95% confidence for numbers near 90-98%, but the paper never states this explicitly; (c) scene/seed variation is fixed to the standard LIBERO evaluation seeds, so cross-paper comparability still assumes identical simulator determinism.

## 3. Statistical Rigor: What the Literature Says

- **How many trials are actually needed.** A Wilson-interval calculation: ±5pp CI at 95% confidence on a single 70% success rate needs **N≈380** rollouts (2605.29710, App. H). Detecting a 5pp *paired* difference between two binary policies at 80% power (McNemar test, α=0.05, discordance rate 0.10–0.25) needs **N≈600–1500 paired rollouts per (model,task) cell** (2605.29710, App. H, citing Connor's formula). The field's modal N=10–20 is roughly **1–2 orders of magnitude under-budget**.
- **Distributional (CDF-based) tests need far fewer trials than binary tests for the same power.** PhAIL's macro-averaged Kolmogorov-Smirnov test on time-to-success CDFs resolves two of three "close" real-VLA pairs (GR00T-N1.6 vs ACT at N=25/cell; OpenPI-π0.5 vs ACT at N=30/cell) where binary success-at-threshold and RMST-as-scalar tests fall well short of 80% detection power across the whole N≤30 range — a claimed **~30× sample-efficiency advantage** over stratified binary McNemar at the same power (2605.29710, §3.3, §5.2).
- **Sequential/anytime-valid testing further cuts sample count**, and richer (non-binary) progress metrics compound the gain. N-SCORE, a safe-anytime-valid-inference sequential test that works on continuous progress scores (not just binary success), shows on the LBM 1.0 dataset: sequential evaluation alone saves 16–25% of trials vs. fixed-batch testing on binary outcomes, and switching from binary to partial-credit progress metrics saves a further **~70% in simulation, ~45% on hardware**, for a combined 24–30% reduction vs. SOTA sequential-binary methods (STEP) (2603.13616, Table II). On RoboArena's crowd-sourced continuous progress-score data, N-SCORE distinguishes all 4 evaluated π0-family policies where a binary-success sequential test (WSR) cannot separate π0 from PG-Diff even after 641 trials (2603.13616, §VI-C.4).
- **Real-to-sim correlation itself needs a statistical vocabulary**, not just "does it look right." SimplerEnv formalizes two metrics for judging any proxy evaluation against ground-truth real trials: Pearson correlation r (rank/trend agreement) and **Mean Maximum Rank Violation (MMRV)** (2405.05941, §III) — reused by AutoEval to show AutoEval (r=0.942, MMRV=0.015) beats SIMPLER (r lower, MMRV higher) as a real-world proxy (2503.24278, §5.2).
- **Crowd-sourced pairwise comparison converges fast but needs its own bias controls.** RoboArena's ranking accuracy (vs. an oracle built from 4284 exhaustive comparisons) saturates within ~100 pairwise episodes — matching the sample-efficiency of conventional fixed-task batch evaluation while giving broader task/scene coverage — but flags Goodhart's-law over-optimization risk once evaluation becomes a training target, and notes it has not tested robustness to adversarial evaluators (2506.18123, §5.3, §7).

## 3b. What a 100-episode sim suite actually resolves (own bed, measured 4–6 Sep 2026, verified)

Numbers from the UR5e sim bed in `santapong/RoboLLM` (`sim/vla-bed`, branch `experiment/ur5e-vla-bed`,
report `results/REPORT-2026-09-06.md`): SmolVLA fine-tuned on 400 scripted demos, scored closed-loop on a
frozen suite of 100 seeded episodes that every recipe replays identically.

- **Unpaired Wilson intervals at n = 100 are ±8 points wide** at success rates of 0.1–0.3; a learning curve of
  0.09 / 0.13 / 0.20 / 0.13 over four checkpoints is flat inside them. **Pairing on identical seeds** (discordant
  pairs, exact McNemar, paired bootstrap) resolved differences of 8–11 points that the marginal intervals
  could not: gain-probe vs nominal +0.11 [+0.04, +0.18], p = 0.007 (11 vs 3 discordant of 100); camera-jitter
  recipe vs its predecessor under a shifted camera +0.08 [+0.01, +0.15], p = 0.057. Two recipes that both score
  0.20 unpaired gave 15 vs 15 discordant pairs (p = 1): "equal" was a real finding, not a failure to resolve.
  This is the cheap end of the 2605.29710 arithmetic above — pairing costs nothing when the evaluator is seeded.
- **A tolerance sweep separates "misses by centimetres" from "goes the wrong way".** Re-thresholding the per-episode
  minimum distance at acceptance radii 0.03 → 0.10 m raised every recipe's success three- to fourfold by 0.06 m
  (0.12 → 0.46 → 0.71 at 0.10 m for the final checkpoint); an out-of-range camera view gave a flatter curve
  (0.06 → 0.17 → 0.42). The curve is right-censored at the suite's own radius (episodes stop at success) and must
  say so. This is the diagnostic 2607.23108 builds its precision scaling law on (failure rate ∝ N^a at fixed
  tolerance; data need grows super-exponentially as the tolerance approaches a system limit c).
- **Success needs a dwell, and the dwell hides a mode.** Counting episodes that *ever* entered the radius
  exceeded the suite's success (five consecutive frames) by 3–12 points; the gap was largest for a policy-side
  clip of the commands (0.19 held vs 0.31 reached), i.e. the clip made the arm reach and jitter out again. A
  binary success metric would have called the clip "no effect".
- **Safety and success decouple.** Capping the demonstration labels below the safety limit removed 32–45 %
  step rejections entirely (safety 0.00 → 0.46–0.99) and changed success by nothing (0.20 → 0.20, p = 1): the
  wrapper had been bounding safety, not success. Report both, never one.
- **A null on the nominal suite can hide a real effect under a variation.** Halving the injected expert
  noise left the nominal learning curve inside the noise at every checkpoint (paired +0.04, −0.04, +0.07,
  +0.01; p 0.19–1) while the same checkpoint under a shifted camera gained +0.16 [+0.08, +0.24] (18 vs 2
  discordant, p = 0.0004) over the azimuth-only recipe and +0.11 [+0.02, +0.20] (p = 0.035) over its own
  nominal score. A single-condition evaluation would have filed the recipe as "no effect"; the variation
  suite, run on the same seeds, is what made the effect visible and attributable.
- **The tolerance sweep predicted which lever would move the ceiling.** Success rose three- to fourfold
  between 0.03 and 0.06 m for every single-camera recipe (centimetre misses, not wrong directions), which
  put the remaining levers on sensing and label clarity rather than demo count. Lower expert noise did
  nothing (above); a wrist camera moved the frozen-suite success from 0.10 to 0.89 with 79 vs 0 discordant
  pairs. A diagnostic that ranks levers before spending compute is worth more than another seed.

## 4. Failure-Detection Taxonomy

| signal source | representative papers | reported detection metric | latency / overhead |
|---|---|---|---|
| **Policy-internal features / hidden state** (supervised probe on last-layer features) | SAFE (2506.09937) | ROC-AUC up to 0.828 (SAFE-MLP) averaged seen+unseen across 4 sim benchmarks; real-world 0.65–0.89 | +0.73ms (SAFE-LSTM, 2.3M params) — negligible vs. π0's 149ms inference |
| **OOD / last-layer Mahalanobis distance + action-chunk consistency** | VLA-FAIL (2606.21386) | AUCPDT (precision+recall+latency combined metric) competitive with/better than 32-sample baselines (ACE, Diff, STAC) at near-zero extra latency | real-time capable — <1ms extra vs 60ms+ for 32-sample baselines |
| **Distributional shift robustness (contrast-set augmented probes)** | SAFECAST (2608.04246) | ROC-AUC gains of +0.10–+0.20 over vanilla SAFE under visual/language/multimodal distribution shift (statistically significant, p<10⁻⁵ to p<10⁻¹⁰ in most cells) | negligible (same probe architecture as SAFE) |
| **Counterfactual attention / KV-cache ablation (functional grounding)** | GUARD (2608.04510) | unseen-task ROC-AUC avg 88.84%, +5.73pp over best competing monitor (FIPER), across Pi0/SmolVLA/Alpamayo × LIBERO/SimplerEnv/MetaWorld/PhysicalAI-AV | 1.14–1.27× inference FLOPs (saliency backward pass + counterfactual probe) |
| **Vision-action spatiotemporal consistency (foveation + verification)** | ActFovea (2607.29169) | success recovery from 49.3%→90.3% under visual overlay (93.7% of gap closed); safe-failure trigger 100% of frozen-replay episodes, 0 unprotected failures | plug-and-play, no retraining |
| **Action-conditioned world-model latents** | Foresight (2606.23085) | ROC-AUC 0.76–0.94 depending on benchmark/policy; best on long-horizon (BEHAVIOR-1K, 8557-step avg rollouts) where baselines drop to 0.54–0.72 | world-model backbone dominates cost: ~183ms/replan-step (queried once per 16-step chunk for π0-FAST) |
| **Attention-entropy (visual-token addressing) / self-evaluation** | FabriMAE / MAE (2608.16697) | AUROC 59–86% depending on policy/subset; +test-time action-selection gain of +1.10pp success on LIBERO-Plus with <0.1× latency overhead | <0.1× overhead (reuses attention already computed) |
| **Failure→correction offline RL** (not detection but recovery) | RedFlow (2607.27782) | +12.0pp avg success on LIBERO (56.2%→68.2%); real-robot 56.7%→74.7%; matches on-policy RL (PPO/GRPO/DDPO) with ~10-24× fewer rollouts | offline, one training pass; no runtime cost |

### 4.1 SAFE's protocol, and the leakage control worth copying (2506.09937)

Fetched in full 2026-09-03. SAFE is already in the table above for its ROC-AUC; what follows is
its *protocol*, which is the more transferable part.

**Setup.** Four policy x benchmark cells, two simulated and two real. LIBERO-10 with OpenVLA,
π0-FAST and π0 on the authors' released checkpoints, no further fine-tuning; reported LIBERO-10
success rates **OpenVLA 53.7%, π0-FAST 60.2%, π0 85.2%** (2506.09937, App. B.5). SimplerEnv with a
π0 reproduction, trained and scored **separately per embodiment** (Google Robot and WidowX have
different feature spaces), 4 tasks each at 100 rollouts; "pick up coke" is excluded because the
policy succeeds 98% of the time and yields no failures to detect. A real Franka running
π0-FAST-DROID over **13 tasks x (30 success + 30 failure)** rollouts. A real WidowX running OpenVLA
over 8 tasks, **532 rollouts, 244 success / 288 failure**.

**The split is the contribution.** The detector trains only on seen tasks and is scored on held-out
ones: 3 of 10 on LIBERO, 1 of 4 per embodiment on SimplerEnv, 3 of 13 on the Franka. Within seen
tasks, 60/40 (LIBERO) or 66/33 (SimplerEnv) train/calibration. Results average over **3 seeds** in
simulation and **5** on real robots, each seed a different seen/unseen split.

**The leakage control (2506.09937, App. B.5).** LIBERO terminates a rollout the moment the task
succeeds, so every *failure* runs to the maximum length. A detector that did nothing but count
elapsed time (s_t = t) would score a perfect ROC-AUC. The authors state this explicitly and fix it
by truncating every rollout in a task to that task's **minimum** rollout length, so success and
failure have identical duration before scoring. **This is the single most reusable idea in the
paper for our purposes** — it is a duration confound that any LIBERO-based failure-detection result
inherits silently, and most papers using LIBERO for this do not mention it.

**Metrics.** Two layers. (1) Threshold-free: ROC-AUC on the **running maximum** score
s̄_t = max_{τ≤t} s_τ evaluated at final T, because a success becomes a false positive if the score
ever crosses the threshold. (2) Thresholded by functional conformal prediction calibrated on
*successful seen-task* rollouts, giving a false-positive-rate guarantee of at most α: TPR, FPR,
balanced accuracy, and **T-det**, the normalized first-crossing timestep averaged over ground-truth
failures (T-det = 1 if never raised). Ground-truth failure timesteps are **hand-annotated** ("when a
human thinks failure happens or intervention is needed"), so early detection is measured against
when a person would have intervened, not against the end of the episode.

**Baseline fairness, stated honestly.** STAC needs many action samples (its own paper uses 256;
SAFE tests it at 10) and is therefore **simulation-only** here: generating 10 samples instead of 1
costs **+152% latency for π0 and +221% for π0-FAST on a single RTX 3090**, which the authors judge
impractical on a real robot. SAFE itself is 2.3M params and +0.73 ms, under 1% of π0's 149 ms.

**Stated limitations.** Manipulation only; no cross-embodiment, sim-to-real, or action-less-video
generalization tested; last-layer features only; and the detector still requires deploying the
policy and collecting both successful and failed rollouts before it can detect anything — so it
does not remove the data-collection cost, it amortizes it across tasks.

## 5. Reactivity vs. Action-Chunking

Action chunking (predict k future actions per inference call) is now near-universal (ACT/ALOHA 2304.13705, π0, SmolVLA, GR00T). It fixes imitation-learning's compounding-error problem — ACT's own ablation shows success rate jumping from **1% at chunk size k=1 to 44% at k=100**, then slightly tapering as k→open-loop (2304.13705, §VI-A) — but at the direct cost of reactivity: mid-chunk, the robot is blind to new observations for k timesteps, and naively switching chunks at the boundary can cause visible strategy-mode jumps ("jerky, out-of-distribution" transitions).

Three mitigations, in order of sophistication:
1. **Temporal ensembling** (ACT, 2304.13705): query the policy every timestep, keep overlapping chunks, and exponentially-weight-average the predictions for the current timestep. Cheap, but this is a *smoothing* operation on predicted actions — RTC's own ablation shows temporal ensembling can produce **invalid actions on multi-modal tasks**, because averaging two valid-but-different strategies is not itself valid (2506.07339, Fig. 5: TE "performs poorly across the board, even with an inference delay of d=0, illustrating the multi-modality of our benchmark — averages of valid actions are not necessarily valid").
2. **Asynchronous inference with a fixed threshold-triggered requery** (SmolVLA, 2506.01844): decouple action *execution* from action *prediction* — start computing the next chunk once the queue drops below a fraction g of chunk size n, instead of waiting for full exhaustion. On real SO-100 tasks this gets ~30% faster task completion at comparable success rate (13.75s→9.7s sync vs async; 9 vs 19 completed pick-place cycles in a fixed 60s window) (2506.01844, §4.6). SmolVLA also finds a **U-shaped tradeoff on chunk size** — n between 10–50 balances reactivity and efficiency; n=1 (no chunking) drops LIBERO avg SR to 50.0%, n=100 drops it to 74.5%, n=10–50 hits 78.5–84.0% (2506.01844, Table 12).
3. **Real-Time Chunking (RTC)** (Physical Intelligence, 2506.07339): the mid-chunk-abort/reactivity problem framed and solved directly as an *inpainting* problem — freeze the prefix of the new chunk that will definitely execute before inference finishes, and generate the rest conditioned on that frozen prefix via a soft-masked flow-matching guidance term. Applicable out-of-the-box to any diffusion/flow VLA, no retraining. On a new 12-task dynamic Kinetix benchmark and on π0.5 real-robot bimanual tasks (light-a-match, plug-ethernet, fold-shirt, etc.), RTC is uniquely robust to injected inference delay — synchronous inference degrades linearly with delay, both temporal-ensembling variants can't run at all at +100/+200ms delay (trigger the robot's protective e-stop from oscillation), while RTC shows **no throughput degradation up to +200ms of injected latency**, and is 20% faster than synchronous baseline even with pauses removed (2506.07339, Fig. 1, Fig. 6). RTC also strictly dominates BID (bidirectional decoding, a rejection-sampling alternative) while using far less compute (BID needs 2.3× RTC's latency at batch=16) (2506.07339, §4.1).

Other 2026 work in the same space that appeared in discovery but was **not fetched** (title/abstract only, not verified): DREAM-Chunk (2606.18589, "Reactive Action Chunking with Latent World Model") and FutureRTC (2607.24008, "Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking") — both extend the RTC idea; numbers not verified here.

## 6. Safety Evals

- **Gemini Robotics / ASIMOV** (2503.20020): introduces the ASIMOV datasets (ASIMOV-Multimodal, visual+text VQA safety scenarios; ASIMOV-Injury, drawn from real NEISS injury records) to test *semantic* action safety — "is this instruction desirable/safe?" rather than physical execution safety. Gemini Robotics-ER + a "safety constitution" (constitutional-AI style rules) reaches **0.88 alignment accuracy** on ASIMOV-Multimodal and **0.88** on ASIMOV-Injury, both up from ~0.85 baseline; performance under adversarial prompting (asked to flip its own judgment) degrades to 0.28 without the constitution and recovers to 0.76 with it (2503.20020, Fig. 29). Notably this is **VQA-level safety reasoning, not closed-loop physical safety** — the model is asked whether an instruction is safe, not evaluated on whether its executed trajectory avoids collisions.
- **LIBERO-Safety** (2606.23686): closed-loop physical-safety benchmark, 19,664 collision-free demonstrations across 40 tasks, 5 hazard/interaction suites (Affordance-Aware Grasping, Human-Robot Interaction, Tabletop Spatial Avoidance, Free-Space Hand-Object Avoidance, Semantic Safety Reasoning), evaluated across 3 difficulty tiers L0-L2. Best model π0.5 tops out at 35.3–88.7% Success Rate depending on suite/tier — collapsing sharply at L2 (OOD conditions). Key finding: **task failure and safety violation are largely decoupled** — models often fail the task (timeout, kinematic deadlock) *without* violating a collision constraint, and separately can violate constraints while completing the task; sub-optimal trajectory synthesis and semantic misalignment (grasping the wrong but visually-similar object) are identified as the two dominant failure modes independent of collision safety (2606.23686, Key Findings 7–8).
- **ForesightSafety-VLA** (2606.27079): 66 base scenarios in RoboTwin, 13-category safety taxonomy (Safe-Core physical / Safe-Lang instruction / Safe-Vis perception), explicitly designed as a **process-level** (not endpoint-only) metric via cumulative safety cost (CC) and risk exposure time (RET), plus a four-quadrant decomposition (safe-success / unsafe-success / safe-failure / unsafe-failure). Central finding: **no evaluated baseline is fully safe** — even the strongest (OpenVLA-oft) has CC=0.18, unsafe-success rate 6%, unsafe-failure rate 15% (2606.27079, Table II). Structure/layout and visual variation degrade safety far more sharply than ordinary language paraphrase, except for adversarial language (prompt injection), which is comparably damaging.
- **LIBERO-VIFO** (2608.17600): a narrower but sharp safety finding specific to visual-cue-conditioned VLAs — when language explicitly conflicts with a visual cue, **0% of 7 tested models followed the unauthorized cue** (language wins), but with **no language instruction at all**, models complete the cue-indicated task **32.2–60.1%** of the time purely from the visual cue — an "unauthorized visual induction" risk. In safety-critical scenes (HazardArena-style hazard/safe task pairs), MolmoAct2 executes the *hazardous* cue-indicated task in 13.3% of authorized-cue episodes.

## 7. Checklist: What a Rigorous VLA Eval Needs

1. **Report N per (policy, task) cell explicitly, and don't trust N<25** — modal current practice is 10-20, and Wilson CIs at that N are ±20-25pp wide (2605.29710, App. H; survey table).
2. **Report confidence intervals on every headline number**, bootstrap or Wilson — only 1 of the 13 surveyed papers did (2605.29710, App. A/Table 5).
3. **Prefer a paired/blinded design over independent evaluation of each policy** — same scene, same operator, randomized order (LBM examination, PhAIL's "single protocol recommendation that does the most work") (2605.29710).
4. **If comparing two close policies, use a distributional test (CDF/KS) or a sequential/SAVI test, not a fixed-N binary test** — both give ~2-30× sample efficiency gains at the same statistical power (2605.29710 §3.3; 2603.13616 §VI).
5. **Use partial-credit / continuous progress scores, not just binary success**, wherever feasible — cuts required trials ~45-70% further on top of sequential testing (2603.13616, Table II) and gives N-SCORE-style tests information binary tests structurally cannot see.
6. **Report the full distribution (time-to-success CDF), not just an aggregate scalar** — different "reasonable" scalar aggregations (success@τ, RMST, AUC-vs-reference) can produce opposite top-1 rankings on identical data when CDFs cross (2605.29710, §5.1).
7. **Log and report spatial/environmental configuration as a confound, not noise** — a same-side vs. opposite-side camera/tote swap moved one policy's completion rate by 22pp, larger than the gap the study was trying to resolve (2605.29710, App. G).
8. **If using simulated evaluation as a proxy, validate real-to-sim correlation with Pearson r AND MMRV**, not "it looks plausible" — validation-loss/MSE is documented to *negatively* correlate with real success (2405.05941, Table I; 2503.24278 confirms).
9. **State whether training-set curation (filtering near-zero actions / failed demos) changed between compared rows** — this alone moved OpenVLA-OFT's reported LIBERO avg from 94.5% to 97.1% (2502.19645, Table I, App. G1).
10. **Report a failure-detection/monitoring signal alongside success rate**, not success rate alone — a policy that is 90% successful but gives no early warning on the 10% failure is a worse deployment candidate than an 85%-success policy with a well-calibrated failure detector (implicit across §4 papers; explicit motivation of SAFE, VLA-FAIL, GUARD).
11. **Distinguish physical-safety violation from task failure as separate axes**, not one conflated metric — LIBERO-Safety shows they're largely decoupled (Key Finding 7/8); ForesightSafety-VLA's four-quadrant decomposition (safe/unsafe × success/failure) is the concrete instantiation.
12. **Measure process-level safety exposure (time spent near a hazard), not just terminal violation** — a trajectory can avoid a hard collision while spending 40-60% of its horizon inside a soft risk buffer; RET catches this where a binary collision flag does not (2606.27079, §V-E, Table IV).
13. **Test under distribution shift for both success AND failure-detection calibration** — a detector calibrated only on in-distribution data degrades under visual/language shift; contrast-set-aware calibration (SAFECAST) recovers a statistically significant fraction of that loss (2608.04246, Table 1).
14. **When claiming "reactive," report degradation vs. injected inference latency explicitly**, not just nominal-latency throughput — synchronous inference and naive temporal ensembling both fail qualitatively (oscillation, protective e-stop) under +100-200ms delay that a real deployed system (remote inference, larger model) will actually encounter (2506.07339, Fig. 6).
15. **Report evaluator/environment reproducibility over time** — AutoEval's two-months-apart reproducibility check (success rates within a few points, reset/success-classifier accuracy still 96%) is the kind of longitudinal check almost no other benchmark reports (2503.24278, App. J).

## 8. Mapping to H5 (Measured-vs-Commanded)

H5's hypothesis: detect policy failure by comparing what the policy **commanded** (predicted action) against what the arm **actually did** (measured via encoder/vision) — a proprioception/command mismatch signal.

**Closest prior art found:**
- **Action Chunk Consistency (ACC), VLA-FAIL (2606.21386)**: measures disagreement between *overlapping predicted action chunks* at consecutive inference steps — i.e., what the policy commands at time t vs. what it re-commands at t+1 for the same future timestep. This is a *commanded-vs-commanded* consistency check, not commanded-vs-executed. It is the single closest existing idea, and the paper explicitly visualizes it as "predicted EEF x-position over episode time" diverging between chunks in failed vs. successful rollouts (2606.21386, Fig. 2) — but it never compares against the robot's actual measured trajectory.
- **STAC (cited inside VLA-FAIL and GUARD, from Agia et al. 2025, "Unpacking failure modes of generative policies: Runtime monitoring of consistency and progress")**: same category — action-chunk-to-action-chunk consistency, not command-vs-execution.
- **ActFovea (2607.29169)**: closest to true measured-vs-commanded, but inverted in direction — it uses *proprioceptive state transitions and robot kinematics* as a reference to judge whether *visual observations* remain consistent with what should have physically happened (i.e., detects stale/corrupted vision, not policy failure per se). Its "dynamic consistency" score explicitly "combines directional and magnitude agreement between predicted and observed image displacement" and includes "action-proprioception agreement" as one input signal to its aggregate risk score (2607.29169, Method section) — this is the one paper found that literally uses proprioception-vs-predicted-action agreement as one ingredient of a failure/risk signal, but it's folded into a broader multi-signal spatiotemporal-consistency score, not isolated and evaluated on its own as a standalone failure detector.
- **RedFlow (2607.27782)**: uses proprioceptive state *q_t* jointly with a learned task-progress estimate to define "execution context" for retrieving corrective targets — proprioception is used for context-matching (what should the arm do next given where it physically is), not as a mismatch/failure signal against the commanded action.
- **GUARD, SAFE, Foresight, FabriMAE, SAFECAST, RedFlow**: all use policy-internal signals (hidden states, attention, KV-cache ablation, world-model latent prediction) or vision, but **none of the papers fetched treat raw proprioceptive-measured-trajectory vs. commanded-action divergence as the primary, standalone failure signal.**

**Verdict: H5 is not directly taken, but is adjacent to two published ideas** — (a) chunk-to-chunk *commanded* consistency (VLA-FAIL's ACC, STAC), which is a weaker proxy than true measured-execution comparison, and (b) ActFovea's proprioception-as-one-of-several-consistency-inputs. The specific, isolated hypothesis — "compare the policy's own predicted joint/EEF trajectory against the encoder-measured trajectory the arm actually achieved, as a standalone failure/anomaly signal, evaluated on its own detection metrics" — was not found as a dedicated paper or ablation in this search. This is a real gap: every failure-detector surveyed either looks *inward* (policy features/attention/latents) or *forward* (world-model predicted future vs. observed future), but not *backward-and-compare* (commanded action vs. what the low-level controller/encoders report actually happened). Given that VLA-FAIL's own ACC and ActFovea's dynamic-consistency component are both partial, weaker versions of this idea and both report it as a useful (if imperfect — ACC "detects failures more reliably" but with delay per VLA-FAIL's own limitations section) signal, H5 has reasonable a priori support without being scooped outright.

## 9. Limitations

- **Discover_papers budget capped at 2 calls per the task instructions** — coverage of the failure-detection and statistical-rigor literatures beyond what those two searches surfaced is not guaranteed complete; there may be additional 2025-2026 papers on measured-vs-commanded or proprioception-based failure signals not surfaced.
- **DREAM-Chunk (2606.18589) and FutureRTC (2607.24008)** appeared in search results but were not fetched — their numbers on reactive chunking are not verified here.
- **Not verified**: whether any paper besides those cited in §8 has directly tested proprioception-vs-command mismatch as an isolated ablation; this is inferred from the absence of such a paper in the two searches and the twelve fetched papers, not from an exhaustive literature check.
- π0/OpenVLA-OFT LIBERO comparison table numbers are current as reported in the respective papers' releases; more recent checkpoints of π0.5/π0.6 may report different LIBERO numbers not checked here.
- The Real-Time Chunking paper's official arXiv id is **2506.07339** (title "Real-Time Execution of Action Chunking Flow Policies"), not literally titled "Real-Time Chunking" — flagged in case the prompt's expected id differed; content matches the requested Physical Intelligence π0 async-inference paper exactly (author Kevin Black, Physical Intelligence, RTC acronym used throughout).
- SAFE's arXiv id is **2506.09937** — confirmed via title-search rather than the exact id string.

## 10. Sources

All fetched via alphaXiv `answer_pdf_queries`; full bib entries in `paper/bib-eval.bib`.

- Curse of Precision (data scaling law for high-precision manipulation) — 2607.23108
- Own UR5e sim bed (paired suites, tolerance sweeps) — github.com/santapong/RoboLLM sim/vla-bed, `results/REPORT-2026-09-06.md`
- LIBERO — 2306.03310
- SimplerEnv — 2405.05941
- CALVIN — 2112.03227
- RoboCasa — 2406.02523
- RoboArena — 2506.18123
- AutoEval — 2503.24278
- OpenVLA-OFT — 2502.19645
- PhAIL — 2605.29710
- Beyond Binary Success (N-SCORE) — 2603.13616
- VLA-FAIL — 2606.21386
- SAFE — 2506.09937
- SAFECAST — 2608.04246
- GUARD — 2608.04510
- ActFovea — 2607.29169
- Foresight — 2606.23085
- FabriMAE (Markov Attention Entropy) — 2608.16697
- RedFlow — 2607.27782
- ACT / ALOHA — 2304.13705
- Real-Time Chunking (RTC) — 2506.07339
- SmolVLA — 2506.01844
- Gemini Robotics / ASIMOV — 2503.20020
- LIBERO-Safety — 2606.23686
- ForesightSafety-VLA — 2606.27079
- LIBERO-VIFO — 2608.17600

Not fetched (title/abstract only from discover_papers, not cited as sourced claims above): DREAM-Chunk (2606.18589), FutureRTC (2607.24008), SafeVLA-Bench (2606.00773), ManiGuard (2608.17386), ROBOGATE (2603.22126), "What Are We Actually Benchmarking in Robot Manipulation?" (2606.04233), "Betting for Sim-to-Real Performance Evaluation" (2604.24018).
