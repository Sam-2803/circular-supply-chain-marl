# Project Cleanup Plan

Phase 1 audit only. No files were deleted, moved, renamed, overwritten, or modified. This plan requires explicit approval before any cleanup action.

## Audit Summary

The project contains:

- A current paper-inspired five-agent MARL implementation.
- A complete test suite covering environment, diagnostics, configuration, observations, actions, KPI measurements, and controlled evaluation.
- Stage 1–6 research evidence and diagnostic outputs.
- A legacy single-agent Q-learning implementation with its own artifacts.
- Three duplicate/snapshot files, two of which are byte-identical copies.
- Root-level generated outputs from earlier legacy experiments.
- A local `.venv` with Python 3.13.4, NumPy, pandas, and Matplotlib.

The current MARL evaluation and evidence depend on files under `marl_artifacts/`; those files should be retained.

## Complete Inventory

### 1. Core implementation: keep

| File | Purpose and dependencies |
|---|---|
| `paper_environment.py` | Current circular-supply-chain digital twin. Used by current MARL, diagnostics, baselines, and tests. Keep. |
| `marl_agents.py` | Five independent linear learners, replay, epsilon-greedy selection, checkpoints. Used by training, diagnostics, and tests. Keep. |
| `marl_training.py` | Current exploratory five-agent training entry point. Writes training logs/checkpoints. Keep. |
| `paper_config.py` | Verified/unresolved paper configuration boundary. Used by No-Op diagnostics and tests. Keep. |
| `marl_observations.py` | Opt-in normalized KPI/full-data/silo-data observation layer. Used by its tests and future paper-alignment work. Keep. |
| `kpi_measurements.py` | Completed-order lead time, delivered-unit cost proxy, and WIP-unavailable diagnostics. Used by No-Op and controlled evaluation. Keep. |
| `noop_normalization.py` | Existing No-Op episode-level diagnostic. Used by tests and README commands. Keep. |
| `noop_period_diagnostic.py` | Stage 6B seed-period panel diagnostic. Writes `marl_artifacts/noop_period_diagnostic_stage6b/`. Keep. |
| `controlled_evaluation.py` | Controlled four-policy evaluation with common random numbers. Writes Stage 4B outputs. Keep. |
| `marl_diagnostics.py` | Training diagnostics and earlier controlled evaluation path; consumes checkpoints and writes `training_100` outputs. Keep until its outputs/workflow are deliberately superseded. |
| `reward_audit.py` | Read-only paper-versus-exploratory reward metadata and validation. Keep. |
| `reward_diagnostics.py` | Read-only reward/action replay diagnostic. Writes Stage 4C outputs. Keep. |
| `validate_environment.py` | Current environment validation across seeds/horizon. Imports `paper_environment.py`. Keep. |

### 2. Tests: keep

| File | Coverage |
|---|---|
| `test_paper_environment.py` | Current paper environment smoke/invariant test. Keep. |
| `test_marl.py` | MARL action/observation/smoke-training checks. Keep. |
| `test_marl_observations.py` | Observation modes, ordering, missing metrics, action encoding. Keep. |
| `test_paper_config.py` | Verified/unresolved configuration boundary. Keep. |
| `test_action_semantics.py` | Current exploratory action mappings and boundaries. Keep. |
| `test_kpi_measurements.py` | Lead time, proxy cost, censoring, zero denominator, reproducibility. Keep. |
| `test_noop_normalization.py` | Existing No-Op diagnostics and provisional normalizer behavior. Keep. |
| `test_noop_period_diagnostic.py` | Stage 6B row count, reproducibility, missingness, ratios. Keep. |
| `test_controlled_evaluation.py` | Horizon, CRN alignment, reproducibility, checkpoint immutability. Keep. |
| `test_reward_audit.py` | Reward audit/reporting boundary. Keep. |
| `test_reward_diagnostics.py` | Reward reconstruction, read-only replay, checkpoint immutability. Keep. |
| `test_supply_chain.py` | Legacy simulator tests. Keep while legacy code/artifacts are retained. Archive only together with the full legacy stack after approval. |

### 3. Research evidence and documentation: keep

| File | Reason |
|---|---|
| `research_paper.pdf` | Primary research evidence. Non-negotiable retention. |
| `README_MARL.md` | Setup, modes, limitations, commands, and stage documentation. |
| `marl_artifacts/stage5_gap_analysis/gap_analysis.md` | Main paper-fidelity gap analysis. |
| `marl_artifacts/stage5_gap_analysis/next_experiment_plan.md` | Controlled future experiment plan. |
| `marl_artifacts/stage5_gap_analysis/noop_normalization_audit.md` | Stage 6A normalization audit. |
| `marl_artifacts/stage5_gap_analysis/reward_specification_evidence.md` | Reward evidence and unresolved paper details. |
| `marl_artifacts/stage5_gap_analysis/stage6d_paper_kpi_definitions.md` | KPI-definition recovery report. |
| `marl_artifacts/stage5_gap_analysis/noop_normalization_audit.md` | Existing normalization data audit. |
| `marl_artifacts/project_cleanup_plan.md` | This approved-before-cleanup plan. Keep as project history. |

Note: the requested `stage6c_normalizer_definition_audit.md` was referenced in a later task but does not exist. It should not be fabricated or listed for deletion.

### 4. Reproducibility artifacts: keep

| Path/file | Why it matters |
|---|---|
| `marl_artifacts/training_100/` | Existing 100-episode training log, five checkpoints, reward/loss plots, parameter diagnostics, evaluation runs, summaries, and scenario comparability. Referenced by `marl_diagnostics.py`, README commands, Stage 4B, Stage 4C, and Stage 5 plans. Keep. |
| `marl_artifacts/smoke/` | Five-agent smoke checkpoints and training log. Demonstrates the initial smoke run. Keep. |
| `marl_artifacts/test/` | Checkpoints/log generated by the unit smoke test. This is disposable test output, but do not delete during this audit; archive or remove only after confirming no report relies on it. |
| `marl_artifacts/controlled_evaluation_stage4b/` | Controlled per-seed results, summary, scenario alignment, metadata. Referenced by Stage 5 analysis. Keep. |
| `marl_artifacts/reward_diagnostics_stage4c/` | Step/episode reward decomposition, action associations, summary, metadata. Referenced by Stage 5 analysis. Keep. |
| `marl_artifacts/noop_period_diagnostic_stage6b/` | 200-row seed-period panel, order panel, candidate summaries, metadata. Required current evidence. Keep. |
| `marl_artifacts/noop_normalization_stage2/` | Earlier No-Op diagnostic before Stage 4A integration. Keep as historical evidence; do not confuse with current schema. |
| `marl_artifacts/noop_normalization_stage4a/` | Expanded No-Op episode outputs with lead time/proxy cost. Keep as historical evidence. |
| `marl_artifacts/stage5_gap_analysis/` | All current evidence and plans. Keep. |

### 5. Potentially redundant root-level generated outputs

These are not imported by the current MARL implementation, but some are consumed by legacy scripts or preserve historical results. They should not be deleted automatically.

| File | Current use/reference | Recommendation |
|---|---|---|
| `evaluation_all_runs.csv` | Read by `plot_results.py`; written by `evaluate_agent.py`. | Keep or archive together with `plot_results.py` and `evaluate_agent.py`; do not delete independently. |
| `evaluation_summary.csv` | Written by `evaluate_agent.py`; historical legacy evaluation. | Archive with legacy stack, not delete yet. |
| `training_results.csv` | Written by `q_learning.py`. | Keep as legacy reproducibility artifact or archive with legacy stack. |
| `q_table_results.csv` | Read by `evaluate_agent.py`; written by `q_learning.py`. | Keep if legacy evaluator is retained; archive as a pair otherwise. |
| `no_op_baseline.csv` | Written by `baseline_evaluation.py` and its identical copy. | Historical current-environment baseline; keep until Stage 4B is declared authoritative, then archive rather than delete. |
| `policy_comparison_raw.csv` | Written by `policy_comparison.py` and its identical copy. | Historical fixed-policy result; archive after confirming no report depends on it. |
| `policy_comparison_summary.csv` | Written by `policy_comparison.py`; historical summary. | Archive with raw CSV and script. |
| `simulation_results.csv` | Legacy simulator result; no current consumer found. | Candidate for archive, not immediate deletion. |
| `strategy_comparison.csv` | Legacy strategy result; no current consumer found. | Candidate for archive, not immediate deletion. |
| `operating_cost_comparison.png` | Historical plot generated by `plot_results.py`. | Archive with `evaluation_all_runs.csv`; not needed by current MARL commands. |
| `emissions_comparison.png` | Historical plot generated by `plot_results.py`. | Archive with legacy plot inputs. |
| `service_level_comparison.png` | Historical plot generated by `plot_results.py`. | Archive with legacy plot inputs. |
| `inventory_comparison.png` | Historical plot generated by `plot_results.py`. | Archive with legacy plot inputs. |
| `__pycache__/` | Python bytecode cache; no source dependency. | Safe to remove only after approval; it is regenerated automatically. |

### 6. Legacy code: retain for now, archive as a coherent set later

| File | Why it is legacy |
|---|---|
| `supply_chain.py` | Older single-agent supply-chain simulator. Imported by `rl_environment.py` and tested by `test_supply_chain.py`. |
| `rl_environment.py` | Older single-agent RL wrapper. Imported by `q_learning.py` and `evaluate_agent.py`. |
| `q_learning.py` | Legacy tabular single-agent Q-learning script; not the paper's five-agent method. Writes root CSVs. |
| `evaluate_agent.py` | Legacy Q-table evaluator; reads `q_table_results.csv`, writes root evaluation CSVs. |
| `test_supply_chain.py` | Tests the legacy simulator. Keep with the legacy stack. |
| `plot_results.py` | Reads legacy `evaluation_all_runs.csv` and writes legacy PNGs. |
| `simulation_results.csv` | Legacy generated result. |
| `strategy_comparison.csv` | Legacy generated result. |
| `training_results.csv` | Legacy Q-learning output. |
| `q_table_results.csv` | Legacy Q-table output. |
| `evaluation_all_runs.csv` | Legacy evaluator output. |
| `evaluation_summary.csv` | Legacy evaluator summary. |
| Root comparison PNGs | Legacy plot outputs. |

Deleting only one legacy file would break its script/test pair or remove reproducibility context. The safe action is to archive the complete legacy group together after approval.

### 7. Environment/configuration: keep

| Item | Assessment |
|---|---|
| `.venv/` | Existing Windows Python environment. It is not source-controlled project logic, but it is required for the documented commands and should not be deleted without a replacement setup/requirements file. |
| `README_MARL.md` | Required setup/configuration documentation. Keep. |
| No `requirements.txt` found | Setup reproducibility is currently tied to `.venv` and README commands. Do not delete `.venv` unless a requirements/environment export is created and verified in a later approved change. |
| No project-level `pyproject.toml` or `setup.py` found | No packaging metadata to clean. |

## Reference and Dependency Findings

### Current MARL path

`paper_environment.py` is used by:

- `marl_training.py`
- `marl_diagnostics.py`
- `controlled_evaluation.py` indirectly
- `noop_normalization.py`
- `noop_period_diagnostic.py`
- `baseline_evaluation.py`
- `policy_comparison.py`
- current environment/action/KPI tests

The trained checkpoint directory is consumed by:

- `marl_diagnostics.py`
- `controlled_evaluation.py`
- `reward_diagnostics.py`
- README commands
- Stage 5 plans

### Duplicate scripts

- `baseline_evaluation.py` and `baseline_evaluation copy.py` are byte-identical.
- `policy_comparison.py` and `policy_comparison copy.py` are byte-identical.
- `paper_environment copy.py` is not byte-identical and is an older snapshot, including older random-state behavior and older order/state logic. It must not be deleted as though it were a harmless duplicate.

### Artifact path risks

The current README and diagnostic metadata refer to exact paths such as:

- `marl_artifacts/training_100/checkpoints`
- `marl_artifacts/controlled_evaluation_stage4b`
- `marl_artifacts/reward_diagnostics_stage4c`
- `marl_artifacts/noop_period_diagnostic_stage6b`

Moving or renaming these requires updating documentation and metadata, or preserving compatibility copies. Deleting them would prevent reproducing the reported Stage 4–6 analyses.

## Recommended Keep Set

Keep all of the following:

- `research_paper.pdf`
- `README_MARL.md`
- Current MARL/core files listed above
- All current tests
- `marl_artifacts/training_100/`
- `marl_artifacts/smoke/`
- `marl_artifacts/controlled_evaluation_stage4b/`
- `marl_artifacts/reward_diagnostics_stage4c/`
- `marl_artifacts/noop_period_diagnostic_stage6b/`
- `marl_artifacts/stage5_gap_analysis/`
- Historical Stage 2/4A No-Op outputs
- `.venv/` until setup reproducibility is replaced and verified

## Recommended Archive Set

Archiving is safer than deletion because it preserves research history:

1. Complete legacy stack:
   - `supply_chain.py`
   - `rl_environment.py`
   - `q_learning.py`
   - `evaluate_agent.py`
   - `plot_results.py`
   - `test_supply_chain.py`
   - legacy CSVs and PNGs listed above
2. Duplicate baseline/policy scripts:
   - `baseline_evaluation copy.py`
   - `policy_comparison copy.py`
3. Older environment snapshot:
   - `paper_environment copy.py`

Archive should be a deliberate approved operation with a manifest, not an ad-hoc move, because root-relative commands and historical outputs currently point at these files.

## Proposed Deletion Set After Approval

The safest initial deletion proposal is intentionally narrow:

1. `__pycache__/` contents: generated bytecode only; regenerated automatically. This is the only clearly disposable item found.
2. After archiving and verifying the legacy group, optionally remove the duplicate byte-identical copies:
   - `baseline_evaluation copy.py`
   - `policy_comparison copy.py`

Do **not** propose deletion of `research_paper.pdf`, current MARL source, tests, checkpoints, Stage 5/6 reports, Stage 6B outputs, or root historical result files during the first cleanup pass.

`paper_environment copy.py` should be archived, not deleted, because it is a distinct historical implementation snapshot.

## Proposed Clean Structure

```text
circular-supply-chain-marl/
├── README_MARL.md
├── research_paper.pdf
├── paper_environment.py
├── paper_config.py
├── marl_agents.py
├── marl_training.py
├── marl_observations.py
├── kpi_measurements.py
├── noop_normalization.py
├── noop_period_diagnostic.py
├── controlled_evaluation.py
├── marl_diagnostics.py
├── reward_audit.py
├── reward_diagnostics.py
├── validate_environment.py
├── tests/                         # optional future relocation; requires import/path updates
├── marl_artifacts/
│   ├── training_100/
│   ├── smoke/
│   ├── controlled_evaluation_stage4b/
│   ├── reward_diagnostics_stage4c/
│   ├── noop_period_diagnostic_stage6b/
│   └── stage5_gap_analysis/
├── archive/
│   ├── legacy_single_agent/
│   ├── duplicate_snapshots/
│   └── historical_outputs/
└── .venv/                         # local environment, retained for now
```

The `tests/` and `archive/` moves are structural proposals only. They should not be performed without updating imports, commands, and a manifest in a separate approved phase.

## Approval Gate

No cleanup action has been taken. Before any file is deleted, moved, renamed, or archived, approval should specify whether to:

- Remove only generated `__pycache__` files.
- Archive the legacy group.
- Archive duplicate/snapshot scripts.
- Remove the two byte-identical copies after archiving.
- Leave all historical root outputs in place.
