# Controlled Next Experiment Plan

This is a proposal only. It does not modify code, checkpoints, reward weights, or training settings.

## Purpose

Discriminate between two explanations for the trained policy's weak service:

- H1: the observed service degradation is primarily caused by the joint action pattern and current dispatch/reserve mechanics.
- H2: the degradation is primarily caused by the provisional reward's multi-objective scalarization and sparse service/OTIF bonuses.

The experiment must not be interpreted as proving either hypothesis without causal design and replication.

## Prerequisites

Before running an alignment experiment, resolve or explicitly freeze:

1. The paper's exact reward terms and No-Op normalizer procedure.
2. The paper's logistics cadence action semantics.
3. The recovered paper KPI weights are recorded, but exact normalizer values,
  KPI orientation, `b_t`, clipping, bonus values, schedule adherence, budget
  conditions, and expediting guard details remain unresolved.
4. Whether the existing exploratory action semantics are being tested as a
  diagnostic control rather than a paper-faithful condition.

If those remain unresolved, run only the diagnostic comparison below and label it exploratory.

## Experiment A: Frozen-policy controlled replay

### Policies

- No-Op
- Random with a fixed policy seed per evaluation seed
- Existing heuristic
- Existing trained checkpoint, greedy/no exploration

### Environment

- Existing `CircularSupplyChainDT`
- No changes to reward or dynamics
- Existing common-random-number streams
- 20-day horizon

### Seeds

Use separate explicit lists:

- Training checkpoint: existing `marl_artifacts/training_100/checkpoints`
- Evaluation seeds: `100, 101, 102, 103, 104, 105, 106, 107, 108, 109`
- Do not use evaluation outcomes to modify training or normalization.

### Metrics

Operational metrics:

- Total operating cost
- Total emissions
- Total waste
- Final backlog
- Final service level
- Final OTIF
- Completed-order average lead time
- Censored order count
- Cost per delivered unit proxy
- WIP unavailable status

Reward diagnostics:

- Reward sum and mean
- Penalty total
- Each individual clipped penalty contribution
- Service bonus count
- OTIF bonus count
- Recovery bonus count
- Schedule-adherence bonus count when the metric is available
- Budget-favorable bonus count when the condition is available
- Expediting-guard projection count and penalty when the guard is available
- Clipping frequency by term

Policy behavior:

- Per-agent action frequencies
- Production input
- Inventory target
- Dispatch quantity
- Transit and expediting decisions
- Recovered material and waste

### Acceptance criteria

This experiment is accepted as a valid diagnostic only if:

- All policies use the same 20-day horizon.
- All policies use identical demand, yield, and latent transport paths per seed.
- Trained weights are unchanged before and after evaluation.
- Reward reconstruction matches environment reward to numerical tolerance.
- Recovered weights are reported separately from unresolved normalizers and
  bonus values.
- Unavailable WIP remains unavailable.
- Proxy cost is never reported as paper-defined cost per unit.
- Mean and sample standard deviation are reported across all ten seeds.

No performance threshold is required because this is an audit, not a claim of improvement.

## Experiment B: Action-pattern sensitivity, exploratory only

This experiment is intended to test the observed association, not to claim causality or paper fidelity.

Use the same ten evaluation seeds and compare frozen policies that alter one action family at a time from the trained action pattern:

1. Trained policy unchanged.
2. Trained planning actions replaced by neutral planning action `1`.
3. Trained inventory actions replaced by neutral inventory action `1`.
4. Trained logistics actions replaced by neutral logistics action `1`.
5. Trained expediting actions replaced by neutral action `0`.
6. Trained recycling actions replaced by neutral recycling action `1`.
7. All action families replaced by the neutral No-Op combination.

This requires a diagnostic policy wrapper only; it must not update weights. The comparison should be paired by seed and report changes in service, OTIF, backlog, reward penalty components, emissions, waste, and cost.

### Interpretation guardrails

- A change after replacing one action family is evidence of sensitivity under the fixed environment, not proof that the agent caused the original outcome.
- Since action interactions are strong, one-at-a-time replacements do not identify independent causal effects.
- Logistics cadence cannot be tested as paper-aligned until its mapping is verified.
- Do not tune reward weights based on this experiment.

## Experiment C: Reward specification readiness check

Do not train. Verify that a future paper-fidelity run cannot start unless the following are supplied:

- Complete KPI weight map.
- Fixed positive normalizer map and seed-period aggregation metadata.
- Additive `b_t` definition.
- KPI orientation transforms.
- Clipping and threshold rules.
- Schedule-adherence and budget-favorable bonus definitions.
- OTIF-shortfall expediting guard threshold, projection, and penalty.
- Exact observation tables.
- Exact action table, especially logistics cadence.
- Discount, replay, epsilon, learning-rate, and update schedule.

The run should fail validation when any required item is absent.

## Expected deliverables

Save future results separately from existing artifacts, for example:

```text
marl_artifacts/stage5_experiment/
  per_seed.csv
  summary.csv
  reward_components.csv
  action_frequencies.csv
  scenario_alignment.csv
  metadata.json
```

Do not overwrite Stage 4B or Stage 4C outputs.
