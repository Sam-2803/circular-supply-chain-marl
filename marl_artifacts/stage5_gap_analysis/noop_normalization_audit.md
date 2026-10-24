# Stage 6A No-Op Normalization Data Audit

No production code or existing artifacts were modified. This audit evaluates whether the current project can implement the recovered paper requirement:

> Estimate KPI normalizers once from No-Op rollouts, average across seeds and periods, then hold the resulting values fixed across experiments.

## Conclusion

The current project can reproduce the No-Op environment and several KPI measurements, but it cannot yet calculate the paper's exact normalizers. Existing outputs contain one aggregate row per seed, not one row per seed-period observation. Several reward KPIs are unavailable or represented by episode aggregates whose aggregation is not equivalent to a period-level average.

The current outputs are suitable for provisional descriptive diagnostics only.

## Paper reward KPIs and current sources

| KPI | Current source | Units | Current definition | Availability |
|---|---|---|---|---|
| CO2 emissions | `metrics["emissions"]` | emissions units per period; current episode diagnostic sums periods | Process plus transport emissions minus recycling savings | Available, but episode aggregation only |
| Operating cost | `metrics["operating_cost"]` | cost units per period; current episode diagnostic sums periods | Production, material, holding, backlog, transport, energy, and recycling costs | Available, but episode aggregation only |
| Inventory | `metrics["inventory"]` | units at period end | Inventory after production and dispatch | Available per period in live `metrics`; Stage 4A stores episode mean |
| Lead time | Order `created_day`, `completed_day` | days per completed order | Completion day minus creation day; incomplete orders censored | Available only for completed orders; no period panel currently saved |
| Material waste | `metrics["net_waste"]` | material units per period | Gross scrap minus recovered material | Available per period in live `metrics`; Stage 4A stores episode total |
| Backlog | `metrics["backlog"]` | outstanding units at period end | Sum of each order's remaining quantity, including in-transit units | Available per period; Stage 4A stores episode mean |
| Cost per unit | No valid paper denominator in current code | unresolved | Current reward uses operating cost divided by production input; paper denominator unresolved | Not paper-available |
| Energy per unit | `energy_used / production_input` | energy per production-input unit | Current diagnostic ratio over episode totals | Available as a ratio, but not period panel |
| Service level | Cumulative on-time units / cumulative demand | unit ratio | Current No-Op output stores final cumulative ratio | Available, but final cumulative value is not a period observation |
| OTIF | Completed OTIF orders / total orders | order ratio | Current No-Op output stores final cumulative ratio | Available, but final cumulative value is not a period observation |
| WIP | None | unresolved | Production-stage WIP is not tracked; pending shipments are not WIP | Unavailable |

Additional paper bonus-related measurements remain unavailable:

- Schedule adherence.
- Budget-favorable regime indicator.
- OTIF-shortfall expediting guard state.

## Existing output schemas

### Stage 2 output

`marl_artifacts/noop_normalization_stage2/noop_kpi_per_seed.csv` contains one row per seed and columns for episode-level values. It does not contain a `period`/`day` column or daily KPI rows. Its `average_lead_time` is unavailable because it predates the Stage 3D measurement integration.

### Stage 4A output

`marl_artifacts/noop_normalization_stage4a/noop_kpi_per_seed.csv` contains one row per seed and adds:

- `average_lead_time`
- `cost_per_delivered_unit` proxy
- `completed_order_count`
- `censored_order_count`

It still does not contain per-period KPI values. `noop_kpi_summary.csv` calculates descriptive statistics across seed-level episode aggregates.

The current `noop_normalization.py` runs daily steps internally but discards the daily `metrics` rows after creating episode aggregates. Therefore the required seed-period panel cannot be reconstructed from the saved CSVs alone.

## Is seed-level averaging equivalent to seed-period averaging?

In general, no.

For an additive period KPI such as operating cost, if every seed has the same horizon $H$, then:

$$
\frac{1}{S}\sum_{s=1}^{S}\left(\sum_{t=1}^{H} c_{s,t}\right)
=
H\left(\frac{1}{SH}\sum_{s=1}^{S}\sum_{t=1}^{H} c_{s,t}\right)
$$

The two quantities differ by a factor of $H$: episode-total mean versus period-level mean. A normalizer must specify which unit is intended.

For average inventory and backlog, the current seed-level episode mean can equal the pooled seed-period mean only when every seed has the same number of valid periods and each seed contributes equally:

$$
\frac{1}{S}\sum_s\left(\frac{1}{H}\sum_t x_{s,t}\right)
=
\frac{1}{SH}\sum_s\sum_t x_{s,t}
$$

This equality holds for the current fixed 20-day horizon when all periods are valid, but it cannot be verified from the saved episode CSV alone.

For ratios such as service level, OTIF, energy per unit, and delivered-unit cost, averaging per-seed ratios is generally **not** equivalent to pooling numerators and denominators across seed-period observations:

$$
\frac{1}{S}\sum_s\frac{N_s}{D_s}
\ne
\frac{\sum_s N_s}{\sum_s D_s}
$$

The current No-Op output stores final cumulative service and OTIF ratios and an episode-total energy ratio. It does not preserve the numerator/denominator columns needed to reproduce a paper-defined seed-period ratio.

For lead time, the current measurement is an average over completed orders per episode. Averaging episode means is not equivalent to pooling all completed orders unless every seed has the same number of completed orders. Censored orders must remain excluded or handled according to the paper; they must never be silently assigned zero lead time.

## Required seed-period normalization panel

A reproducible diagnostic dataset should contain one row per `(seed, period)` and, where needed, separate order-level completion data.

### Required period-panel schema

```text
seed
period
horizon
scenario
policy                     # No-Op
exogenous_demand
production_input
usable_output
yield_rate
operating_cost
emissions
net_waste
energy_used
inventory_end
backlog_end
delivered_units
on_time_delivered_units
cumulative_demand
cumulative_on_time_units
total_orders_to_date
completed_orders_to_date
otif_orders_to_date
pending_shipments
baseline_transit_days
schedule_adherence         # unavailable until defined
budget_favorable           # unavailable until defined
wip                        # unavailable until tracked
```

### Required order-level table for lead time and OTIF

```text
seed
order_id
created_day
due_day
completed_day
delivered_quantity
required_quantity
completed_by_due
```

### Aggregation formula

For each KPI, first define its period-level value and units. Then, for seeds $s=1,\ldots,S$ and periods $t=1,\ldots,H$:

1. Run the No-Op policy with fixed seed $s$ and horizon $H=20$.
2. Store every valid period observation.
3. Exclude unavailable or undefined values; do not replace them with zero.
4. Compute the paper-specified average over the valid seed-period panel:

$$
K_k^{norm} = \frac{1}{N_k}\sum_{(s,t)\in V_k} K_{k,s,t}
$$

where $V_k$ is the valid observation set and $N_k=|V_k|$.

5. Record the valid count, excluded count, exclusions/reason, units, and aggregation formula.
6. Estimate once and freeze the resulting map for the entire evaluation/training experiment.

This is the natural implementation of the recovered “averaged across seeds and periods” wording. It must remain explicitly paper-comparable only until the exact paper definition of each period KPI is confirmed.

For additive quantities, use period values directly. Do not sum them before calculating a period normalizer unless the paper explicitly defines an episode-total KPI. For ratios, preserve numerators and denominators and use the paper's specified ratio aggregation; do not silently average ratios.

## Invalid and unavailable values

- `NaN`, missing, undefined, or censored values are excluded from the valid set and counted in metadata.
- Zero denominators produce `NaN` plus a reason such as `zero_delivered_units` or `zero_production_input`.
- Negative values are not silently accepted for KPIs that must be nonnegative. They should be rejected or recorded as invalid with the seed/period and reason.
- WIP remains unavailable until production-stage work is explicitly tracked.
- Paper cost per unit remains unavailable until its denominator is recovered.
- Schedule adherence and budget-favorable status remain unavailable until their definitions are implemented.
- No missing KPI is substituted with zero.
- A normalizer must be finite and strictly positive before it can be frozen. If the valid set is empty or its statistic is nonpositive, the paper-fidelity run must fail validation.

## Recommended fixed configuration

For the next normalization diagnostic, use:

```text
horizon = 20 days
policy = No-Op
seeds = 100, 101, 102, 103, 104, 105, 106, 107, 108, 109
aggregation = explicit seed-period mean for additive period KPIs
ratio handling = preserve numerators/denominators and follow paper definition
normalizers = estimated once, saved with schema/units/counts, then frozen
```

The existing Stage 2/4A seed sets are suitable for smoke checks but are not sufficient evidence for a final paper-comparable normalizer protocol, especially because they do not preserve the period panel and use only two or three seeds.

## Implementation plan (future stage)

1. Add a new diagnostic output function, without changing reward or training, that saves every No-Op daily metrics row with `seed`, `period`, and schema metadata.
2. Add an order-level diagnostic export for lead time and OTIF.
3. Add explicit numerator/denominator columns for service level, OTIF, energy per unit, and delivered-unit cost proxy.
4. Add a reducer that validates units, missingness, nonnegativity, denominators, and valid counts.
5. Save candidate normalizers and an immutable metadata file; do not auto-activate them in `paper_environment.py`.
6. Add tests for pooled versus per-seed aggregation, unequal valid counts, zero denominators, censored orders, deterministic seeds, and frozen reuse.
7. Only after paper KPI definitions are fully confirmed should a paper-faithful reward configuration consume the frozen map.

## Final status

The current artifacts cannot recover exact paper normalizers. They provide useful exploratory episode-level references, but not the required seed-period normalization panel. A future implementation must preserve period observations, order-level lead-time events, ratio numerators/denominators, units, valid counts, and exclusions before any normalizer can be labelled paper-comparable.
