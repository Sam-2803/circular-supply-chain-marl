# Reward Specification Evidence Audit

Date: 2026-10-24

This report is read-only. No production code, reward formula, checkpoints, or training settings were modified.

## PDF extraction status

The workspace virtual environment has no installed PDF extraction library (`pypdf`, `PyPDF2`, `pdfplumber`, `fitz`, and `pdfminer` were unavailable). No command-line extractor was available either.

A direct binary search of `research_paper.pdf` found no plain-text occurrences of `omega`, `No-Op`, `bonus`, `clipping`, or `OTIF`; it found only isolated compressed/object occurrences for `normalizer` and `reward`. Therefore the PDF text layer cannot recover the numerical reward table.

The integrated PDF viewer was inspected visually around pages 8–12. Page 9 supports the general reward equation and normalization notation, but the numerical table/coefficient text is not readable at the available rendering quality. Values below are never inferred from the current implementation.

## Recovered paper specification

### Equation 7, PDF page 9

The readable equation is:

$$
r_t=-\sum_{k\in K}\omega_k\bar K_{k,t}+b_t
$$

with normalized KPI notation:

$$
\bar K_{k,t}=\frac{K_{k,t}}{K_k^{norm}}
$$

The equation establishes:

- A shared scalar reward.
- A negative weighted sum of normalized KPI terms.
- One coefficient $\omega_k$ per KPI.
- A positive normalizer notation $K_k^{norm}$.
- An additive term $b_t$.

Additional recovered paper text supplies the eight KPI weights and the
normalizer estimation procedure. The exact numerical normalizer values and
the full additive-bonus implementation remain unresolved.

### KPI scope, PDF pages 8–9

The surrounding KPI/state discussion visibly includes the following paper-relevant measures:

- Average lead time.
- Service level / OTIF.
- Operating cost.
- Cost per unit.
- CO2 emissions.
- Energy per unit.
- Material waste.
- Average inventory.
- WIP.
- Backlog.

The paper states that KPI normalizers are estimated once from No-Op rollouts,
averaged across seeds and periods, and held fixed across experiments. The
current artifacts do not yet implement that exact seed-period panel
aggregation for every KPI.

The paper also states that bonus triggers include service level at least 0.95,
OTIF at least 0.95, schedule adherence at least 0.90, low-waste regimes, and
budget-favorable regimes. Exact bonus values and some thresholds remain
unresolved. Expediting has an OTIF-shortfall guard: when it binds, the action
is reduced to the largest admissible level and a small penalty is applied. The
exact guard threshold and penalty remain unresolved.

The objective maximizes discounted episodic return with $\gamma$ approximately
1. The exact selected value is not sufficiently precise to replace the current
exploratory setting.

## Comparison table

| Component | Paper specification | Current implementation | Evidence | Status | Required action |
|---|---|---|---|---|---|
| Reward structure | $r_t=-\sum_k\omega_k\bar K_{k,t}+b_t$ | Negative weighted penalty sum plus provisional bonuses | PDF page 9, Eq. 7 | VERIFIED structure; implementation not exact | Preserve exploratory mode; obtain readable coefficients before adding paper mode |
| KPI normalization | $\bar K_{k,t}=K_{k,t}/K_k^{norm}$ | Manual constants in `paper_environment.py` | PDF page 9 normalization notation | PARTIALLY VERIFIED | Recover units and No-Op aggregation |
| KPI weights | CO2 1.1; operating cost 1.2; inventory 0.5; lead time 0.7; material waste 1.2; backlog 0.6; cost per unit 1.0; energy per unit 0.6 | Same numerical weights in the exploratory reward | Recovered paper text; Eq. 7/page 9 | VERIFIED values; implementation still provisional overall | Preserve values, but verify KPI orientation/normalizers before paper mode |
| CO2 | KPI included; lower is naturally consistent with negative penalty, but exact orientation is not readable | `emissions / 5000`, clipped at `2.0`, weight `1.1` | Pages 8–9 KPI discussion; current environment | PARTIALLY VERIFIED | Confirm paper orientation and normalizer |
| Operating cost | KPI included | `operating_cost / 15000`, clipped at `2.0`, weight `1.2` | Pages 8–9; current environment | PARTIALLY VERIFIED | Confirm paper aggregation and normalizer |
| Inventory | KPI included | `inventory / 600`, clipped at `2.0`, weight `0.5` | Pages 8–9; current environment | PARTIALLY VERIFIED | Confirm average/ending/excess definition |
| Lead time | KPI included | Daily `transit_days / 3`, clipped at `2.0`, weight `0.7` | Pages 8–9; current environment | PARTIALLY VERIFIED | Confirm whether paper means order lead time, average lead time, or transit proxy |
| Waste | KPI included | `net_waste / 20`, clipped at `2.0`, weight `1.2` | Pages 8–9; current environment | PARTIALLY VERIFIED | Confirm gross/net definition and normalizer |
| Backlog | KPI included | `backlog / 1500`, clipped at `2.0`, weight `0.6` | Pages 8–9; current environment | PARTIALLY VERIFIED | Confirm daily/average/terminal aggregation |
| Cost per unit | KPI included | `(operating_cost / production_input) / 30`, clipped at `2.0`, weight `1.0` | Pages 8–9; current environment | PARTIALLY VERIFIED | Recover paper denominator and units |
| Energy per unit | KPI included | `(energy_used / production_input) / 2`, clipped at `2.0`, weight `0.6` | Pages 8–9; current environment | PARTIALLY VERIFIED | Confirm denominator and normalizer |
| Service level | Bonus trigger at service level ≥ `0.95`; exact bonus value and any continuous KPI term unresolved | `+0.5` when service level ≥ `0.95` | Recovered paper text; current environment | PARTIALLY VERIFIED | Recover exact bonus value and orientation/weight |
| OTIF | Bonus trigger at OTIF ≥ `0.95`; exact bonus value unresolved | `+0.5` when OTIF ≥ `0.95` | Recovered paper text; current environment | PARTIALLY VERIFIED | Recover exact bonus value and orientation/weight |
| Schedule adherence | Bonus trigger at schedule adherence ≥ `0.90` | No schedule-adherence metric or bonus | Recovered paper text | UNRESOLVED/mismatch | Define and measure schedule adherence before implementation |
| Low-waste regime | Bonus trigger exists; exact regime/value unresolved | Recovery-ratio bonus `+0.25` at `0.4` | Recovered paper text; current environment | PARTIALLY VERIFIED/mismatch risk | Recover low-waste definition and bonus value |
| Budget-favorable regime | Bonus trigger exists; exact condition/value unresolved | No budget-favorable bonus | Recovered paper text | UNRESOLVED/mismatch | Recover budget condition and value |
| Additive term $b_t$ | Explicitly present in Eq. 7 | No named $b_t$; current code uses service/OTIF/recovery bonuses | PDF page 9, Eq. 7 | UNRESOLVED/mismatch | Recover exact $b_t$ definition |
| Clipping | No clipping rule recovered in the additional evidence | Every current normalized value is clipped at `2.0` | Current environment only | UNRESOLVED/mismatch risk | Confirm whether paper clips and at what bound |
| Recovery bonus | Not recoverable from readable paper evidence | `+0.25` if recovered/gross scrap ≥ `0.4` | Current environment only | UNRESOLVED/provisional | Do not label as paper behavior |
| Terminal reward adjustment | No separate terminal bonus recovered | No environment reward terminal adjustment; terminal TD target is handled in learner | Eq. 7 plus recovered text | UNRESOLVED | Confirm terminal reward treatment |
| No-Op normalizers | Estimated once from No-Op rollouts, averaged across seeds and periods, then fixed | Stage 2 computes per-seed episode aggregates and averages across seeds; it does not build a uniform seed-period panel or activate values | Recovered paper text; `noop_normalization.py` | PARTIALLY VERIFIED | Add a paper-specific frozen seed-period aggregation only after KPI definitions are complete |
| Discount factor | Approximately $\gamma=1$ | Exploratory config uses `0.95`; paper config leaves it unresolved | Recovered paper text; `paper_config.py` | PARTIALLY VERIFIED | Recover exact selected value/precision before changing training |
| Expediting guard | OTIF-shortfall guard reduces action to largest admissible level and applies small penalty | No guard or action projection; transit divisor is applied directly | Recovered paper text; current environment | UNRESOLVED/mismatch | Recover guard threshold, penalty, and admissibility rule |

## Current implementation comparison

### `paper_environment.py`

The current exploratory reward contains eight manually normalized lower-is-better penalty terms:

```text
co2              weight 1.1, normalizer 5000, clip 2.0
cost             weight 1.2, normalizer 15000, clip 2.0
inventory        weight 0.5, normalizer 600, clip 2.0
lead_time        weight 0.7, normalizer 3, clip 2.0
waste            weight 1.2, normalizer 20, clip 2.0
backlog          weight 0.6, normalizer 1500, clip 2.0
cost_per_unit    weight 1.0, normalizer 30, clip 2.0
energy_per_unit  weight 0.6, normalizer 2, clip 2.0
```

It adds:

- `+0.5` for service level at least `0.95`.
- `+0.5` for OTIF at least `0.95`.
- `+0.25` when recovered/gross scrap is at least `0.4`.

The numerical weights now agree with recovered paper evidence, but the current
normalizers, clipping, orientations, and bonus values remain exploratory and
are not sufficient for a paper-faithful reward.

### `reward_audit.py`

`reward_audit.py` correctly records:

- Page 9 as the source page.
- Equation 7's general form.
- The current exploratory weights and manual normalizers.
- Current clipping and bonus behavior.
- Unresolved normalizer values/units, $b_t$, orientation, clipping, exact bonus values, schedule adherence, budget condition, expediting guard, and terminal details.

It intentionally refuses to construct a paper-faithful reward.

### `reward_diagnostics.py`

`reward_diagnostics.py` reconstructs the current exploratory reward from environment metrics, including:

- Raw normalized values.
- Clipped values.
- Weighted penalty contributions.
- Service, OTIF, and recovery bonuses.
- Clipping flags.
- Reconstructed reward.

It does not recover or implement the paper reward.

### `paper_config.py`

`paper_comparable_config()` leaves reward weights, No-Op normalization, and $b_t$ unset. This is correct given the available evidence. `exploratory_config()` retains the current provisional settings explicitly.

## Exact missing evidence required

To recover the exact paper reward, provide a clearer copy or screenshot of:

1. Exact numerical values and units for every $K_k^{norm}$.
2. The complete formula for $b_t$, including service, OTIF, schedule-adherence, low-waste, and budget-favorable bonuses.
3. KPI orientation transforms for service/OTIF and any other benefit KPI.
4. Any clipping rule and bound.
5. Exact schedule-adherence definition and metric measurement rule.
6. Exact OTIF-shortfall expediting guard threshold, action projection, and penalty.
7. Terminal reward handling and the precise discount value around $\gamma\approx1$.

Until this evidence is available, the paper reward is only verified at the structural level. The current project must retain its exploratory reward and must not claim exact reproduction.
