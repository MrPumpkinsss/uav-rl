# Diverse Top-K experiment (archived)

This experiment tested MMR-style diversity selection over PPO candidate
assignments. It was validated on the common 32-channel/4-noise true-CodeLlama
set and did not improve the original Top-K policy, so it is not a current
runtime path.

The source is retained for reproducibility only. The active policy remains the
original `scripts/ppo/train_layerwise_topk.py` and its standard Top-K inference.
