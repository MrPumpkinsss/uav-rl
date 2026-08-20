# Current command-line entry points

This directory intentionally contains only the commands needed for current work.

| Path | Purpose | Typical command |
| --- | --- | --- |
| `benchmarks/measure_ppl.py` | Measure the latency of complete true CodeLlama PPL evaluation before a long run. | `python scripts/benchmarks/measure_ppl.py --device cuda --dtype bfloat16` |
| `ppo/train.py` | Train or losslessly resume PPO using true PPL rewards; retained as the direct-oracle baseline. | `python scripts/ppo/train.py --run-dir artifacts/runs/ppo/<run-name> --episodes 1000 --device cuda` |
| `ppo/train_surrogate.py` | Behavior-clone the default strong-link policy, then train or losslessly resume PPO with a frozen surrogate only. It exports frozen candidates every 200 episodes and performs zero true-PPL calls. | `python scripts/ppo/train_surrogate.py --run-dir artifacts/runs/surrogate_ppo/<run-name> --episodes 1000 --device cuda` |
| `ppo/validate_true_policy.py` | Select frozen surrogate-PPO candidates with independent real-model validation; never feeds labels back into surrogate training. | `python scripts/ppo/validate_true_policy.py --run-dir artifacts/runs/surrogate_ppo/<run-name> --device cuda` |
| `ppo/compare_true_baselines.py` | Compare the true-validated policy with common baselines on the same already consumed validation channels/seeds. | `python scripts/ppo/compare_true_baselines.py --run-dir artifacts/runs/surrogate_ppo/<run-name> --device cuda` |
| `surrogate/extend_labels.py` | Resume the current targeted surrogate label extension; research only, not a PPO entry point. | `python scripts/surrogate/extend_labels.py --device cuda` |
| `surrogate/train.py` | Train and validation-select the current global surrogate ensemble; research only, not a PPO entry point. | `python scripts/surrogate/train.py --device cuda` |
| `surrogate/train_residual.py` | Train or exactly resume the targeted tail residual ensemble. It saves state after every epoch and never loads final-test data or starts PPO. | `python scripts/surrogate/train_residual.py --device cuda` |
| `surrogate/diagnose_coverage.py` | Measure label precision, training-feature coverage, and frozen validation support without true-PPL calls or retraining. | `python scripts/surrogate/diagnose_coverage.py --device cuda` |
| `maintenance/verify_legacy_archive.py` | Verify hashes and file sizes of the original archive. | `python scripts/maintenance/verify_legacy_archive.py` |

Set `PYTHONPATH=src` before running scripts from a source checkout.

## PPO run lifecycle

Use a fresh descriptive run directory for a new experiment. The training script refuses to overwrite any existing run artifact.

```powershell
python scripts/ppo/train.py `
  --run-dir artifacts/runs/ppo/true-ppl-round-01 `
  --episodes 1000 `
  --device cuda
```

To extend it, reuse exactly the same directory and change only the total episode target:

```powershell
python scripts/ppo/train.py `
  --run-dir artifacts/runs/ppo/true-ppl-round-01 `
  --episodes 3000 `
  --device cuda `
  --resume
```

Historical surrogate scripts are deliberately absent from this directory. The active surrogate commands have narrowly defined roles: extending high-variance labels, training the global ensemble, and training its tail residual correction. All remain separate from the real-PPL PPO path, and neither training command starts final evaluation or PPO automatically.

## Surrogate PPO warm start

`ppo/train_surrogate.py` uses `strong_link` behavior cloning by default (4,096
synthetic channels and 30 epochs), then fine-tunes exclusively with frozen
surrogate reward relative to that teacher, with a small online behavior-cloning
anchor to prevent unproductive structural drift. Each candidate at episodes 200, 400, 600, and so on is
immutable and must be selected only with a separate true-PPL validation run.
Use `--teacher dynamic_programming` to warm start from DP, or `--teacher none`
to run an intentionally random-start ablation. The teacher choice, cloning
settings, surrogate hash, and candidate interval are saved in the immutable
resume metadata, so a resumed run cannot silently change its training setup.

For a lower-cost policy-selection workflow, run an independent screening stage
over all candidates with `--stage-name screening_validation`, then use its top
candidates as repeated `--candidate` values in a new confirmation stage with a
different `--stage-name`, channel seed, and noise seed. Stage-specific JSON,
selected checkpoint, and PPL cache files prevent screening evidence from being
mistaken for confirmation evidence.
