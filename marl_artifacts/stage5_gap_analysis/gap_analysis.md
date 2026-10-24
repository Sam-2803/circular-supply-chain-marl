# Stage 5 Paper-Fidelity Gap Analysis

Generated from the readable evidence in `research_paper.pdf`, the current source tree, and Stage 4B/4C artifacts. No production code, reward weights, environment dynamics, agent weights, or checkpoints were modified.

## Executive Summary

The project is a functioning exploratory implementation with five independent linear learners, a shared scalar reward, seeded common-random-number evaluation, and diagnostic coverage. It is not a faithful reproduction of the paper because the exact KPI feature tables, reward weights/normalizers/additive term, complete action mappings, replay/training settings, and several environment semantics remain unresolved.

The highest-priority fidelity issue is the reward implementation details that
remain unresolved after the KPI weights were recovered, followed by feature
specification, logistics action semantics, and the training protocol.
Algorithmic tuning should wait until those contracts are verified.

## Paper Requirements Versus Implementation

| Requirement | Paper evidence | Current implementation | Confidence | Remaining gap |
|---|---|---|---|---|
| Five decision agents | Pages 8-10 describe planning, inventory, logistics, expediting, and recycling agents | `AGENT_NAMES` contains five independent agents | High | Exact per-agent observation/action tables remain unresolved |
| Independent learning | Page 10 describes independent learners with separate linear action-value functions | `LinearDQNAgent` instance per agent; separate weights and replay | High | Exact replay/update schedule is unresolved |
| Shared reward | Pages 9-10 describe a shared global reward | All agents receive `metrics["reward"]` | High | Current reward formula is provisional, not paper-equivalent |
| Linear action value | Page 10 gives a linear action-value function with fixed feature map and one-hot action encoding | Existing learner uses one weight row per action and dot product; opt-in `marl_observations.py` can append one-hot action features | High | Default trainer still uses the older provisional observation builder |
| One-step TD target | Pages 10-11 show terminal reward target and discounted max-next-action target | `train_step()` implements this form | High | Exact replay and batch/update timing unresolved |
| Greedy deployment | Page 10 gives greedy argmax deployment | Controlled evaluation uses `explore=False` | High | No major gap in this behavior |
| Horizon | Page 12 shows a 20-day paper-comparable horizon | Controlled Stage 4B uses 20 days; exploratory defaults remain 60 days | High | Paper-comparable mode is not wired into training automatically |
| Learning rate | Page 12 visibly suggests an approximate range near 3e-4 to 3e-3 | Exploratory learner uses 0.01 | Medium/High | Exact selected paper value must be recovered before changing it |
| KPI feature map | Pages 8-10 describe normalized KPI snapshots plus operational context and action encoding | Optional layer supports normalized available KPIs, context, and one-hot action; silo sets are provisional | High | Exact per-agent features and Full/Silo access rules unresolved |
| No-Op normalization | Normalizers estimated once from No-Op rollouts, averaged across seeds and periods, then fixed | Stage 2 measures per-seed episode aggregates and averages across seeds; values are not activated | High | Exact period-level panel, units, and unavailable-KPI handling still need a paper-specific implementation |
| Reward equation | Page 9 gives `r_t = -sum_k(omega_k * normalized_KPI_k,t) + b_t` | Exploratory reward uses eight manually normalized/clipped penalties and three bonuses | High | Weights recovered; normalizer values, orientation, `b_t`, clipping, exact bonus values unresolved |
| Recovered KPI weights | CO2 1.1; cost 1.2; inventory 0.5; lead time 0.7; waste 1.2; backlog 0.6; cost per unit 1.0; energy per unit 0.6 | Current exploratory weights match these values | High | Matching weights do not establish full reward fidelity |
| Bonus triggers | Service ≥0.95, OTIF ≥0.95, schedule adherence ≥0.90, low-waste and budget-favorable regimes | Only service, OTIF, and recovery-ratio bonus are present | High | Exact values/definitions and missing triggers unresolved |
| Discount objective | Discounted episodic return with gamma approximately 1 | Exploratory config uses gamma 0.95 | Medium/High | Exact selected paper gamma must be recovered before changing training |
| Expediting guard | OTIF-shortfall guard reduces action to largest admissible level and applies small penalty | No guard/projection/penalty exists | High | Threshold, admissibility, and penalty unresolved |
| Planning action | Page 9 visibly supports discrete production/planning levels including approximately 0.9/1.0/1.1 | Uses 0.9/1.0/1.1 multipliers | Medium/High | Full table and exact semantics should be verified |
| Inventory action | Page 9 supports inventory-level decisions | Uses -100/0/+100 target adjustments | Medium | Exact values/operational rule unresolved |
| Logistics action | Page 9 describes consolidation cadence, including daily versus every-two-day behavior | Uses 0.75/0.90/1.00 dispatch-capacity fractions | High | Confirmed semantic mismatch; complete paper action table required |
| Expediting action | Page 9 supports expediting/urgency choices | Uses transit divisors 1.0/1.3/1.5 | Medium | Exact factors, variance, cost, and action meanings unresolved |
| Recycling action | Page 9 supports recovery-intensity choices | Uses 0.2/0.4/0.6 recovery fractions | Medium | Exact values and material-quality constraints unresolved |
| Feasibility layer | Page 8 describes a safety/feasibility layer | Environment clips capacity and validates action IDs internally | Medium | Paper projection rules and event logging are not implemented |
| Lead time | Paper KPI discussion includes average lead time | Diagnostic computes completed-order completion day minus creation day; censored orders counted | Medium | Paper aggregation and end-of-horizon treatment unresolved |
| Cost per unit | Paper KPI discussion includes cost per unit | Paper-defined denominator unavailable; delivered-unit proxy is separately labeled | High | Exact paper denominator and aggregation unresolved |
| WIP | Paper KPI discussion includes WIP | WIP remains unavailable; pending shipments are not substituted | High | Production-stage WIP state tracking is required |
| Evaluation | Page 12 describes repeated fixed-seed experimental blocks | Stage 4B uses explicit seeds and common random numbers; reports mean/std | High | Full paper benchmark, robustness, ablation, and confidence protocol not reproduced |

## Logistics Mismatch

The paper evidence on page 9 describes logistics decisions in terms of shipment-consolidation cadence, visibly including daily versus every-two-day consolidation. The current implementation instead computes:

```text
raw_dispatch_capacity = transport_capacity * dispatch_fraction[action]
```

with fractions `0.75`, `0.90`, and `1.00`.

This is not a harmless naming difference: cadence changes when shipments are released, while fractions change the maximum quantity released in a dispatch event. A faithful implementation requires the paper's complete action table, including:

- The number of logistics actions.
- The exact cadence associated with each action.
- Whether cadence means daily/every-two-day release, consolidation windows, or thresholds.
- How orders arriving during a consolidation window are grouped.
- Whether transport capacity is applied per dispatch day or per consolidation period.
- How expediting interacts with a consolidated shipment.
- Whether fixed transport cost and emissions are charged per shipment or per unit.

Until those details are readable, the current dispatch-fraction mode must remain exploratory.

## Why Trained Service Is Weak

Stage 4B/4C evidence supports the following observations, without establishing causality:

- Trained mean reward: approximately `-103.91`, better than heuristic `-106.46`, No-Op `-115.16`, and random `-115.23`.
- Trained mean penalty total: approximately `108.91`, lower than heuristic `111.46` and both other baselines.
- Trained service level: approximately `0.375`, versus heuristic `0.752`.
- Trained OTIF: approximately `0.167`, versus heuristic `0.583`.
- Trained final backlog: approximately `1,743`, versus heuristic `1,055`.
- Trained emissions and waste are lowest among the evaluated policies.
- Trained policy receives no service or OTIF bonuses in the three-seed diagnostic, but receives the recovery bonus on every episode.
- Trained actions are concentrated: inventory action `0`, logistics action `2`, recycling action `2` on all 20 steps; planning is mostly action `0`; expediting is mostly action `1`.

The current reward can prefer the trained policy despite weak service because it sums multiple lower-is-better penalties, clips each normalized penalty at `2.0`, and grants a recovery bonus. Service and OTIF are threshold bonuses rather than smooth reward terms. A policy can therefore reduce emissions/waste and retain a lower aggregate penalty while never reaching the service/OTIF thresholds.

These are measured associations. They do not prove that any single agent caused the service degradation. The likely interaction to investigate first is the joint combination of planning, inventory reserve, and logistics dispatch decisions, because those jointly determine available stock for dispatch. The diagnostic evidence does not justify changing reward weights before paper terms are verified.

## Prioritized Fidelity Work

### P0: Resolve before algorithm tuning

1. Recover exact normalizer values/units, KPI orientation, `b_t`, clipping, exact bonus values, schedule adherence, budget condition, terminal rules, and the expediting guard.
2. Recover the per-agent feature tables and Full-Data/Silo-Data access rules from the paper.
3. Recover the complete logistics cadence action table and dispatch cost semantics.
4. Recover exact replay, discount (approximately gamma 1), epsilon, learning-rate selection, and update schedule from the training table/pseudocode.

### P1: Implement after P0 evidence is available

1. Add a paper-fidelity configuration that refuses unresolved values.
2. Replace only the provisional feature map in an opt-in paper-comparable trainer.
3. Implement logistics cadence as a separate opt-in mode with unit tests.
4. Implement paper No-Op normalizers as a frozen evaluation configuration.
5. Add explicit feasibility projection and requested-versus-executed action logging.

### P2: Expand validation

1. Add settling-period or censoring rules matching the paper.
2. Add WIP state tracking if the paper requires it.
3. Re-run controlled baselines with the same exogenous paths.
4. Perform paper-specified robustness, ablation, and information-regime experiments.
5. Only then tune or extend training.

## Limitations

- The local PDF renders but does not provide reliable machine-readable extraction for several small tables and coefficients.
- Stage 4B used three evaluation seeds, so uncertainty estimates are preliminary.
- Stage 4C action/outcome summaries are observational associations, not causal estimates.
- Current rewards, action semantics, and default observations remain exploratory.
- No claim of exact paper reproduction is supported.
