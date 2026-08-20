# Historical archive

Nothing in `legacy/` is a current training entry point. It is retained for provenance, reproducing an older experiment, or auditing archived results.

## Contents

| Directory | Contents | Current replacement |
| --- | --- | --- |
| `prototype_v0/` | Original PPO prototype, old environment, checkpoints, logs and TensorBoard runs. | `src/uav_rl/` plus `scripts/ppo/train.py` |
| `surrogate_v1_fixed_seed/` | Early fixed-noise-seed PPL data and single-model surrogate CLI scripts. | No current surrogate training entry point; direct true-PPL PPO is the active path. |
| `experiments_2026_08/ppo/` | Completed PPO context reconstruction and surrogate-PPO benchmark scripts. | `scripts/ppo/train.py` |
| `experiments_2026_08/surrogate/` | Completed multi-seed surrogate v2–v5 data generation, training, auditing, diagnostics and scheduled launch scripts. | Retained for future research only; not approved for PPO reward training. |

## Archive integrity

`archive_manifest_2026-08-16.json` records archived paths, original and archived SHA256 values, file sizes and reasons for the initial archive. Verify it from the repository root:

```powershell
python scripts/maintenance/verify_legacy_archive.py
```

Large experiment outputs are under `artifacts/archive/`. Existing multiseed PPO and surrogate research artifacts were intentionally left in their original `artifacts/` locations so previous reports remain reproducible.

External Qwen3/28-layer projects and their data are outside this archive and were not copied, moved or registered.
