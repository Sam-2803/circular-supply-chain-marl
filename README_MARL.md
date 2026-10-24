# Paper-Inspired Cooperative MARL

This module is separate from the legacy single-agent implementation in
`q_learning.py`, `rl_environment.py`, and `supply_chain.py`.

## Configuration modes

`paper_config.py` records the audit status of each setting without changing
the existing training commands.

- **Exploratory mode** uses the current provisional assumptions, including the
  60-day default, learning rate `0.01`, discount factor `0.95`, replay values,
  epsilon schedule, and current environment reward/action mappings.
- **Paper-comparable mode** records the verified five-agent architecture and a
  20-day horizon. It deliberately leaves unresolved parameters unset.
- **Fully paper-faithful mode** is not available yet. Calling
  `require_paper_faithful()` raises `UnresolvedPaperSettingError` until the
  unresolved paper details are confirmed.

Use `exploratory_config()` for the existing implementation and
`paper_comparable_config()` to inspect the verified paper metadata. Neither
function changes the current trainer automatically.

## Verified from the paper

- Five cooperative decision areas: planning, inventory, logistics,
  expediting, and recycling.
- Five independent agents with a shared global reward.
- Linear DQN-style action-value learning.
- One-step TD targets with terminal handling.
- Greedy evaluation policy.
- Paper-comparable episode horizon: 20 days.
- Approximate visible learning-rate range: `3e-4` to `3e-3`.

The local PDF renders correctly but does not provide usable text extraction.
Exact local-observation tables, replay settings, optimizer details, and target
network scheduling could not be verified reliably. They are implementation
assumptions in this smoke implementation.

## Unresolved paper details


The existing six-feature observations, three-action mappings, replay capacity,
batch size, discount factor, learning rate, epsilon schedule, and reward
constants remain available only as explicitly provisional exploratory settings.

## Stage 2: No-Op KPI diagnostic

The No-Op diagnostic measures candidate reference values over explicitly
supplied fixed seeds using the paper-comparable 20-day horizon:

```powershell
.\.venv\Scripts\python.exe .\noop_normalization.py --seeds 100 101 102 103 104 --output-dir marl_artifacts\noop_normalization
```

It writes:

  sample count, availability, and KPI definitions

These are **candidate/provisional** No-Op references only. The diagnostic does
not alter the reward, select normalizers automatically, or claim that the
paper's unresolved aggregation procedure has been recovered. The optional
`select_candidate_normalizers()` function requires an explicit statistic and
labels its result provisional; selected values are intended to remain fixed
within one evaluation run.

Currently unavailable from the environment are average lead time, WIP, and a
paper-defined cost-per-unit denominator. Energy per unit is derived explicitly
as total energy divided by total production input.


## Stage 6B: No-Op seed-period diagnostic

The standalone `noop_period_diagnostic.py` records a 20-period No-Op panel for
explicit seeds `100` through `109`:

```powershell
.\.venv\Scripts\python.exe .\noop_period_diagnostic.py --output-dir marl_artifacts\noop_period_diagnostic_stage6b
```

Outputs:

- `noop_period_panel.csv`: one row per `(seed, period)` with raw measurements,
  cumulative numerators/denominators, derived ratios, units/status fields, and
  explicit unavailable metrics.
- `noop_order_panel.csv`: order creation, due, completion, quantities,
  censoring, and completed-order lead time.
- `candidate_normalizers.csv`: candidate/provisional reductions with the
  aggregation method and valid observation count; values are not activated.
- `diagnostic_metadata.json`: seeds, horizon, schema limitations, and output
  definitions.

The diagnostic preserves raw fields separately from derived ratios. Additive
KPIs use period means; service/OTIF and energy use ratios of aggregated
numerators and denominators; lead time uses a pooled completed-order mean.
Cost per unit, WIP, schedule adherence, and budget-favorable status remain
unavailable. These values are not paper-faithful normalizers until the exact
paper KPI definitions and aggregation rules are verified.
## Stage 3A: Observation audit and opt-in feature layer

`marl_observations.py` provides a separate observation builder for experiments
that supply a fixed `NormalizerConfig`. It supports three explicit modes:

- `exploratory`: no paper KPI features; use the existing `marl_agents.py`
  observation function for the unchanged exploratory trainer.
- `full_data`: all available normalized KPI features plus operational context.
- `silo_data`: provisional agent-specific KPI access sets, because the exact
  paper silo tables remain unresolved.

Missing lead time, WIP, cost-per-unit, or missing normalizers are reported in
`ObservationResult.missing_metrics`; they are not replaced with zero. The
module also provides `action_feature_map()` for the paper's normalized-feature
plus one-hot-action representation. This is an opt-in layer and is not wired
into the default trainer.

The current silo access sets are implementation proposals, not paper-verified
observation tables. No candidate No-Op normalizer is selected automatically.

## Stage 3B: Action-semantics audit

The current exploratory action mappings are:

| Agent | Current mapping | Paper evidence | Status |
|---|---|---|---|
| Planning | `0/1/2 -> 0.9/1.0/1.1` production multiplier | Page 9 visibly supports discrete planning levels including these multipliers | Provisional implementation, strongest match |
| Inventory | `0/1/2 -> -100/0/+100` inventory-target adjustment | Page 9 supports inventory-level decisions, but exact values are not readable | Provisional |
| Logistics | `0/1/2 -> 0.75/0.90/1.00` dispatch-capacity fraction | Page 9 describes shipment-consolidation cadence, including daily versus every-two-day consolidation | Confirmed semantic mismatch |
| Expediting | `0/1/2 -> 1.0/1.3/1.5` transit-time divisor | Page 9 supports urgency/expediting, but exact factors are unresolved | Provisional |
| Recycling | `0/1/2 -> 0.2/0.4/0.6` recovery fraction | Page 9 supports recovery-intensity decisions, but exact values are unresolved | Provisional |

The paper's readable evidence does not provide a complete, reliable mapping for
all action IDs. In particular, it is not safe to invent the missing logistics
levels or silently replace dispatch fractions with cadence behavior. Therefore
the existing dispatch-fraction mode remains the default and no optional
paper-aligned action mode is introduced in Stage 3B. The logistics cadence is
recorded as an unresolved implementation gap until its complete action table
can be verified.

The regression tests in `test_action_semantics.py` pin the current exploratory
mapping and its valid boundaries. They do not claim that these mappings are
paper-faithful.

## Stage 3C: Reward audit

The readable paper evidence on page 9 specifies the shared reward structure:

```text
r_t = -sum_k(omega_k * normalized_KPI_k,t) + b_t
normalized_KPI_k,t = KPI_k,t / K_k^norm
```

The paper indicates positive No-Op-derived normalizers, but the exact KPI
weights, units, benefit/cost orientation transform, additive term `b_t`,
clipping, threshold bonuses, and terminal adjustments remain unresolved.

The current exploratory reward in `paper_environment.py` is unchanged and is
explicitly provisional. It penalizes normalized CO2, operating cost,
inventory, lead time, waste, backlog, cost per unit, and energy per unit using
weights `1.1`, `1.2`, `0.5`, `0.7`, `1.2`, `0.6`, `1.0`, and `0.6`. Its manual
normalizers are `5000`, `15000`, `600`, `3`, `20`, `1500`, `30`, and `2`, with
each term clipped at `2.0`. It adds provisional bonuses for service level at
`0.95`, OTIF at `0.95`, and recovery ratio at `0.4`.

`reward_audit.py` reports this comparison without calculating or changing the
environment reward. `require_paper_reward_details()` intentionally fails until
the unresolved paper evidence is recovered. A paper-faithful reward cannot be
implemented until the exact page-9 weights, No-Op aggregation/units, `b_t`,
KPI orientation, clipping, and bonus/terminal rules are available.

## Stage 3D: KPI measurements

`kpi_measurements.py` measures only KPIs supported by tracked events:

- **Average lead time:** completed-order `completed_day - created_day`, in days.
  Orders incomplete at the episode boundary are censored and excluded; their
  count is reported separately.
- **Cost per delivered unit:** total operating cost divided by total delivered
  units. This is an explicit operational proxy, not the paper's unresolved
  cost-per-unit denominator.
- **WIP:** unavailable. Production-stage WIP is not tracked, and pending
  shipments are not silently treated as WIP.

The environment records order creation and completion days for diagnostics
only. Production, dispatch, shipment timing, rewards, observations, and
training are unchanged. Empty episodes and zero denominators return `NaN` with
explicit status/count fields rather than fabricated values.

Stage 4A integrates these measurements into `noop_normalization.py`. The
per-seed CSV now adds `average_lead_time`, `cost_per_delivered_unit`,
`completed_order_count`, and `censored_order_count`, while retaining the
existing `cost_per_unit` and `wip` columns as explicitly unavailable. The
summary CSV includes rows for the same fields. The delivered-unit measure is
marked as a proxy and is excluded from candidate paper normalizer selection;
completed and censored counts are diagnostic fields, not KPI normalizers.

Focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest .\test_kpi_measurements.py
```

## Smoke test

From PowerShell:

```powershell
.\.venv\Scripts\python.exe .\marl_training.py --episodes 5 --episode-length 10 --seed 42 --output-dir marl_artifacts\smoke
.\.venv\Scripts\python.exe -m unittest .\test_marl.py
```

The run writes one `.npz` checkpoint per agent and `training_log.csv` under
`marl_artifacts\smoke`. These are smoke-test artifacts, not paper results.

## Stage 4B: Controlled baseline evaluation

Use the separate controlled evaluator for paper-comparable, inference-only
comparisons. It defaults to a 20-day horizon when invoked explicitly and uses
the same supplied seeds for No-Op, random, heuristic, and trained policies:

```powershell
.\.venv\Scripts\python.exe .\controlled_evaluation.py --seeds 100 101 102 103 104 --checkpoint-dir marl_artifacts\training_100\checkpoints --output-dir marl_artifacts\controlled_evaluation_stage4b --episode-length 20
```

Outputs:

- `evaluation_per_seed.csv`: one row per policy and seed
- `evaluation_summary.csv`: mean and sample standard deviation across seeds
- `scenario_alignment.csv`: demand, yield, and latent transport alignment checks
- `evaluation_metadata.json`: seeds, horizon, policy definitions, inference mode,
  metric notes, and limitations

The trained checkpoints are loaded for greedy inference only; evaluation does
not call replay or parameter updates. `reward_sum` and `reward_mean` are kept
separate from operational metrics. `cost_per_delivered_unit` is an explicitly
labeled proxy, and WIP remains unavailable. Existing evaluation artifacts are
not overwritten by this command.

## Stage 4C: Reward and action diagnostics

The training log contains aggregate reward, loss, parameter norms, and epsilon,
but not individual reward terms or actions. The read-only diagnostic replays
existing checkpoints and baselines without training:

```powershell
.\.venv\Scripts\python.exe .\reward_diagnostics.py --seeds 100 101 102 --checkpoint-dir marl_artifacts\training_100\checkpoints --output-dir marl_artifacts\reward_diagnostics_stage4c --episode-length 20
```

It writes step-level reward decomposition, clipping flags, bonuses, actions,
and operational outcomes, plus per-episode and per-policy summaries. The
action/outcome tables are associations only and do not establish causality.
The current reward remains provisional and is reconstructed without changing
the environment or checkpoint weights.
