# Archived segment PPO modules

The contiguous-segment policy family was superseded by the current arbitrary
layer-to-UAV policy. Its trainer, policy implementation, and regression test are
kept here only for historical reproduction.

Archived files:

- `segment_policy.py`
- `segment_ppo.py`
- `test_segment_ppo.py`

New experiments should use `scripts/ppo/train_layerwise_topk.py`.
