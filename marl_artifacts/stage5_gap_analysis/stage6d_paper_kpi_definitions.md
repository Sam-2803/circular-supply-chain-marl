# Stage 6D: Paper KPI Definition Recovery

This is a read-only investigation. No Python files, reward logic, agents, training, checkpoints, or Stage 6B outputs were modified.

The requested `stage6c_normalizer_definition_audit.md` was not present in the workspace. The available predecessor audit was `stage5_gap_analysis/noop_normalization_audit.md`; this report uses that audit plus direct visual inspection of `research_paper.pdf`.

## Paper evidence recovered

- Pages 8–9 list the KPI/state scope used by the paper: average lead time, OTIF/service, operating cost, cost per unit, CO2, energy per unit, material waste, average inventory, WIP, and backlog.
- Page 9, Equation 7 gives the shared scalar reward:

  $$
  r_t=-\sum_{k\in K}\omega_k\bar K_{k,t}+b_t
  $$

  with normalized KPI notation:

  $$
  \bar K_{k,t}=K_{k,t}/K_k^{norm}.
  $$

- Recovered text associated with the reward section gives the KPI weights: CO2 `1.1`, operating cost `1.2`, inventory `0.5`, lead time `0.7`, material waste `1.2`, backlog `0.6`, cost per unit `1.0`, and energy per unit `0.6`.
- The paper states that normalizers are estimated once from No-Op rollouts, averaged across seeds and periods, and held fixed across methods, scenarios, and ablations. Page 12 also describes sector-specific normalizers and a two-stage weight-selection procedure.
- Exact KPI equations, numerator/denominator definitions, units for all KPI terms, incomplete-order treatment, zero-denominator rules, and several bonus formulas were not recoverable from the local visual rendering.

## Compact comparison

| KPI | Paper definition/evidence | Current `paper_environment.py` / Stage 6B definition | Status |
|---|---|---|---|
| Operating cost | KPI named in pages 8–9 and normalized in Eq. 7; exact temporal aggregation/unit formula not readable | Sum of daily `operating_cost` in episode diagnostics; Stage 6B stores daily `metrics["operating_cost"]` and candidate mean across 200 periods | Plausible but provisional |
| CO2 emissions | KPI named in pages 8–9; exact emissions boundary/formula not readable | Daily `metrics["emissions"]`: process plus transport emissions minus recycling savings; Stage 6B period mean | Plausible but provisional |
| Inventory | KPI named in pages 8–9; exact average/ending/excess definition not readable | `metrics["inventory"]` at period end; Stage 6B averages `inventory_end` | Plausible but provisional |
| Material waste | KPI named in pages 8–9; exact gross/net/recovered convention not readable | `metrics["net_waste"] = gross_scrap - recovered`; Stage 6B period mean | Plausible but provisional |
| Backlog | KPI named in pages 8–9; exact treatment of in-transit units and temporal aggregation not readable | `_backlog()` sums order `remaining`, including in-transit demand; Stage 6B averages `backlog_end` | Plausible but provisional |
| Energy per unit | KPI named in pages 8–9; exact denominator not readable | Stage 6B uses `sum(energy_used) / sum(production_input)` | Plausible but provisional; denominator unresolved |
| Cost per unit | KPI named in pages 8–9; exact numerator/denominator and units not readable | Current reward uses operating cost divided by production input, but the paper KPI is marked unavailable; no Stage 6B candidate | Unavailable |
| Service level | Service/OTIF KPI scope and bonus threshold `service >= 0.95` recovered; exact KPI numerator/denominator not readable | Stage 6B stores period `on_time_delivered_units / new demand` and candidate total ratio `sum on-time units / sum demand`; period ratio can exceed 1 because deliveries may fulfill earlier backlog | Plausible but provisional; current period definition is not validated by paper |
| OTIF | OTIF scope and threshold `OTIF >= 0.95` recovered; exact denominator and completion-window rule not readable | Stage 6B stores period `otif completions / new orders`; candidate total `sum OTIF orders / sum new orders`; order panel preserves due/completion data | Plausible but provisional |
| Average lead time | KPI named in pages 8–9; exact order cohort, censoring, and horizon treatment not readable | Order panel uses `completed_day - created_day`; censored orders excluded; candidate pooled mean over completed orders | Plausible but provisional |
| WIP | KPI named in pages 8–9; no readable equation or state definition | Not tracked; `NaN`; pending shipments are not substituted | Unavailable |
| Schedule adherence | Recovered reward text identifies a bonus trigger at `>= 0.90`; definition/equation not readable | Not measured | Unavailable |
| Budget-favorable status | Recovered reward text identifies a budget-favorable bonus regime; condition not readable | Not measured | Unavailable |

## Numerator and denominator audit

### Service level

Current Stage 6B fields:

- Numerator: `on_time_delivered_units`, calculated from the increment in `env.total_on_time_units` during the period.
- Denominator: `demand_numerator`, calculated from the increment in `env.total_demand` during the period.
- Candidate total: sum of period numerators divided by sum of period denominators.

The environment increments on-time units when an arriving allocation satisfies `self.day <= order["due_day"]`. Because an arriving shipment may fulfill an older order, same-period on-time deliveries can exceed same-period new demand. The resulting period ratio above 1 is evidence that this is not a conventional new-demand service-rate definition without further paper support.

**Paper status:** numerator and denominator unresolved. The recovered paper evidence does not specify whether service is cumulative fill rate, period demand fulfillment, order-level service, or another measure.

### OTIF

Current Stage 6B fields:

- Numerator: `otif_orders` incremented when an order is fully delivered by its due day.
- Period denominator: `new_orders`, i.e. orders created in that period.
- Candidate total: total OTIF completions divided by total new orders.

An order can be created in an earlier period and completed later, so `otif_orders / new_orders` by period is a cohort-mismatched ratio unless the paper explicitly defines it that way. The separate order panel preserves `created_day`, `due_day`, `completed_day`, required quantity, delivered quantity, and `completed_by_due`.

**Paper status:** completion rule is plausible from the environment but not paper-verified. Exact denominator/cohort and end-of-horizon treatment remain unresolved.

### Energy per unit

Current Stage 6B fields:

- Numerator: `energy_used` per period.
- Denominator: `production_input` per period.
- Candidate normalizer: sum of all period energy divided by sum of all period production input.

This is internally consistent with the current simulation, but the paper's “unit” could refer to production input, usable output, delivered units, or another denominator.

**Paper status:** denominator unresolved.

## Temporal aggregation audit

The recovered paper wording supports a normalizer estimated once from No-Op rollouts and averaged across seeds and periods. Stage 6B correctly preserves a 200-row panel and marks candidates inactive.

Current candidate methods:

- Operating cost, CO2, inventory, waste, backlog: arithmetic mean over 200 period observations.
- Energy per unit: ratio of aggregated totals.
- Service level: ratio of aggregated on-time units to aggregated period demand numerators.
- OTIF: ratio of aggregated OTIF completions to aggregated new-order counts.
- Lead time: pooled mean over completed orders.

These are defensible candidate methods for an audit, but the paper does not visibly specify which KPI terms are period means versus ratios of totals versus order-level pooled measures. The report therefore does not label any candidate as paper-faithful.

## Availability and missingness

Stage 6B handles missingness conservatively:

- Cost per unit: `NaN`, paper denominator unresolved.
- WIP: `NaN`, production-stage WIP is not tracked.
- Schedule adherence: `NaN`, definition and measurement unavailable.
- Budget-favorable status: `NaN`, condition unavailable.
- Lead time: only completed orders contribute; censored orders remain explicitly marked.
- Zero denominators are represented as `NaN` with status/reason fields in the diagnostic logic.

No unavailable KPI is replaced with zero.

## Current versus paper

The Stage 6B panel is a sound reproducible data capture layer, but it does not resolve the paper definitions. In particular:

1. The period service ratio is not bounded and should not be called paper service level without clarification.
2. The period OTIF ratio uses new orders as denominator while completions can belong to earlier cohorts.
3. Lead time is order-level, not naturally period-level.
4. Energy per unit uses production input, which is a current-code choice rather than a recovered paper denominator.
5. WIP, cost per unit, schedule adherence, and budget-favorable status remain unavailable.
6. Operating cost, emissions, waste, inventory, and backlog are period measurements, but the paper's exact temporal scope and normalizer units remain unspecified.

## Unresolved questions requiring author clarification or better evidence

- Is service level cumulative on-time unit fill, period demand fulfillment, order-level service, or another measure?
- Should service numerator include deliveries fulfilling prior-period backlog?
- Is OTIF denominated by all orders, orders due in the period, newly created orders, or completed orders?
- How are incomplete/censored orders handled at the episode boundary?
- Is lead time averaged over completed orders only, all orders with censoring treatment, or a period aggregate?
- Is energy per unit divided by production input, usable output, delivered units, or another quantity?
- Is cost per unit based on production, usable output, delivered units, or another denominator?
- Is inventory an end-of-period level, time average, excess above target, or another KPI?
- Is backlog measured including in-transit quantities as the current environment does?
- Are operating cost, emissions, waste, inventory, and backlog normalized as period means, episode totals, or scenario-specific aggregates?
- What exact state variables define WIP?
- What is schedule adherence and how is its `>= 0.90` bonus trigger computed?
- What defines a budget-favorable regime?
- What normalizer statistic and units are used per KPI, and are sector-specific normalizers always required?

## Final classification

- **Supported by paper evidence:** KPI categories, Equation 7 structure, recovered KPI weights, No-Op rollout normalization concept, averaging across seeds and periods, and holding normalizers fixed across experiments.
- **Plausible but provisional:** Stage 6B period means for additive current-code metrics, pooled completed-order lead time, aggregated energy ratio, and aggregated service/OTIF ratios.
- **Unavailable:** paper-defined cost per unit, WIP, schedule adherence measurement, budget-favorable condition, exact service/OTIF definitions, exact energy/cost denominators, and exact temporal aggregation rules for each KPI.
