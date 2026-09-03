# Recent LLM system baselines

This document fixes the implementation and reporting contract for the recent
LLM-oriented baselines. The root README contains the concise version; this file
is the detailed reference used when modifying experiments.

## EdgeShard-UAV

`src/uav_rl/system_baselines/edge_shard_uav.py`

- **Action family:** an ordered subset of UAVs; every selected UAV appears once
  and owns one contiguous Transformer-layer shard.
- **State:** `(next_layer, used_uav_mask, previous_uav)`.
- **Transition:** choose an unused UAV and the end of its next contiguous shard.
- **Selection objective:** analytical computation latency plus boundary
  communication latency. It does **not** query the PPL surrogate.
- **Constraints:** shard memory and compute energy are checked during expansion;
  exact shared-bandwidth communication energy and all repository constraints are
  checked on terminal deployments.
- **Robust reranking:** the DP keeps the lowest `K` partial plans per state, then
  terminal plans are reranked with `layerwise_latency`, which uses the project's
  exact optimal shared-bandwidth latency formula.
- **Qualification:** this is an adaptation of EdgeShard's device-selection and
  contiguous-sharding principle to dynamic UAV links, not its original runtime.

## HexGen-inspired search

`src/uav_rl/system_baselines/hexgen_search.py`

- **Action family:** ordered, non-repeating UAV pipeline plus variable contiguous
  shard boundaries.
- **Initialization:** EdgeShard-UAV plus reproducible random feasible pipelines.
- **Fitness:** the same frozen surrogate reward used to screen PPO deployments.
- **Mutation:** shift a shard boundary, swap two UAVs, or replace one UAV by an
  unused UAV. Every candidate is checked against exact memory and energy limits.
- **Selection:** elitist evolutionary search; no online training occurs.
- **Qualification:** this borrows topology-aware heterogeneous plan search from
  HexGen, but does not reproduce tensor parallelism, serving runtime, or request
  scheduling. It must always be called **HexGen-inspired**.

Because HexGen-inspired reads the surrogate during search while EdgeShard-UAV
does not, decision time and selector information must be reported beside reward.

## Exact grouped oracle

`src/uav_rl/system_baselines/exact_grouped_oracle.py`

- Divide 32 adjacent layers into `G` deterministic near-equal super-layers.
- Enumerate all `U^G` assignments, including assignments where a UAV reappears.
- Expand every assignment to the original 32-layer action.
- Filter using exact memory, computation-energy, communication-energy and hover
  constraints.
- Score every feasible assignment with the common environment in batches.
- Return the highest-reward assignment and enumeration counts.

This result is exact **only inside the grouped action set**. For five UAVs and
eight groups it evaluates `5^8 = 390625` assignments per channel. It is not a
full `5^32` oracle and must not be presented as one. With the surrogate evaluator
it is also not true-LLM evidence.

## Frozen evaluation rule

The screening script stores `channels.npy` and `frozen_deployments.npz`. True-LLM
evaluation must consume those exact files. It is forbidden to rerun EdgeShard,
HexGen-inspired, PPO, or another selector after inspecting true PPL.
