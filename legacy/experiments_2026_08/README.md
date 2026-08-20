# Completed experimental scripts

These scripts supported surrogate research that has concluded. They are grouped here so that `scripts/` remains a short list of current commands.

## `ppo/`

- `reconstruct_ppo_training_context.py`: reconstructed context for an older cached PPO run.
- `benchmark_true_policy_ppl.py`: evaluated an older policy/baseline setup that depended on a surrogate oracle.
- `train_and_evaluate_ppo.py`: older surrogate-reward PPO experiment.

## `surrogate/`

This directory contains the full historical v2–v5 workflow: multi-seed data collection, tail data extensions, ensemble and gated-expert training, diagnostics, output audits and PowerShell scheduling helpers.

The final seed24 gated-expert development result improved tail prediction substantially, but did not pass the overall and tail MAE acceptance thresholds. Therefore these scripts must not be used to start a new PPO acceptance run without a new surrogate validation plan.

Historical scripts retain their original names because those names are part of recorded experiment provenance. They are not maintained as current command-line interfaces.
