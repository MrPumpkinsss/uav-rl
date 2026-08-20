"""Paper-style resource-constrained layer-to-UAV assignment primitives.

Unlike the legacy fixed-eight-layer setting, a UAV may own any subset of
layers and may reappear after another UAV. Feasibility follows the paper's
memory and energy constraints; communication, PPL noise, and latency are paid
at every cross-UAV layer boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from uav_rl.config import SystemConfig
from uav_rl.wireless import packet_drop_probability


def _layer_memory_profile(num_layers: int) -> tuple[float, ...]:
    """Return a documented research-default memory profile in units."""

    return tuple(0.9 + 0.2 * ((layer % 5) / 4.0) for layer in range(num_layers))


def _layer_compute_profile(num_layers: int) -> tuple[float, ...]:
    """Return per-layer seconds at unit compute speed for the research scenario."""

    return tuple(0.018 + 0.004 * ((layer % 7) / 6.0) for layer in range(num_layers))


@dataclass(frozen=True)
class ResourceConstrainedConfig:
    """Memory/energy parameters of the paper's general assignment model.

    Defaults are a *research scenario*, not measured hardware specifications.
    They deliberately prevent any one UAV from storing all 32 layers and are
    written into every dataset/training manifest for replacement by a measured
    profile later. The base wireless parameters remain in `SystemConfig`.
    """

    system: SystemConfig = field(default_factory=SystemConfig)
    layer_memory_units: tuple[float, ...] | None = None
    layer_compute_seconds_at_unit_speed: tuple[float, ...] | None = None
    activation_mbit_by_boundary: tuple[float, ...] | None = None
    uav_memory_capacity_units: tuple[float, ...] = (8.5, 13.5, 10.5, 16.5, 12.5)
    # The previous draft left only ~1.2 J after hover, which made the planned
    # high-boundary coverage (12--15 transitions) infeasible.  These budgets
    # retain heterogeneous constraints while covering that intended range.
    uav_energy_budget_joule: tuple[float, ...] = (4.80, 4.90, 4.70, 5.00, 4.85)
    uav_hover_energy_joule: tuple[float, ...] = (2.00, 2.00, 2.00, 2.00, 2.00)
    compute_energy_coefficient: float = 0.45

    def __post_init__(self) -> None:
        layers = self.system.num_layers
        uavs = self.system.num_uavs
        if self.layer_memory_units is None:
            object.__setattr__(self, "layer_memory_units", _layer_memory_profile(layers))
        if self.layer_compute_seconds_at_unit_speed is None:
            object.__setattr__(
                self,
                "layer_compute_seconds_at_unit_speed",
                _layer_compute_profile(layers),
            )
        if self.activation_mbit_by_boundary is None:
            object.__setattr__(self, "activation_mbit_by_boundary", (4.0,) * (layers - 1))
        assert self.layer_memory_units is not None
        assert self.layer_compute_seconds_at_unit_speed is not None
        assert self.activation_mbit_by_boundary is not None
        if len(self.layer_memory_units) != layers:
            raise ValueError("layer_memory_units must contain one value per layer")
        if len(self.layer_compute_seconds_at_unit_speed) != layers:
            raise ValueError("layer_compute_seconds_at_unit_speed must contain one value per layer")
        if len(self.activation_mbit_by_boundary) != layers - 1:
            raise ValueError("activation_mbit_by_boundary must contain one value per boundary")
        for values, name in (
            (self.uav_memory_capacity_units, "uav_memory_capacity_units"),
            (self.uav_energy_budget_joule, "uav_energy_budget_joule"),
            (self.uav_hover_energy_joule, "uav_hover_energy_joule"),
        ):
            if len(values) != uavs:
                raise ValueError(f"{name} must contain one value per UAV")
        if min(
            *self.layer_memory_units,
            *self.layer_compute_seconds_at_unit_speed,
            *self.activation_mbit_by_boundary,
            *self.uav_memory_capacity_units,
            *self.uav_energy_budget_joule,
            *self.uav_hover_energy_joule,
            self.compute_energy_coefficient,
        ) <= 0.0:
            raise ValueError("all resource values must be positive")
        if np.sum(self.layer_memory_units) > np.sum(self.uav_memory_capacity_units):
            raise ValueError("total UAV memory is insufficient for all layers")
        if any(
            budget <= hover
            for budget, hover in zip(
                self.uav_energy_budget_joule, self.uav_hover_energy_joule, strict=True
            )
        ):
            raise ValueError("each UAV energy budget must exceed its hover energy")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceUsage:
    """Per-UAV resource use induced by one general layer assignment."""

    memory_units: np.ndarray
    computation_energy_joule: np.ndarray
    communication_energy_joule: np.ndarray
    total_energy_joule: np.ndarray


@dataclass(frozen=True)
class ResourceLatency:
    """Computation and optimally shared-bandwidth communication latency."""

    computation_seconds: float
    communication_seconds: float
    total_seconds: float


def validate_layerwise_deployment(
    deployment: np.ndarray,
    config: ResourceConstrainedConfig,
    *,
    channel: np.ndarray | None = None,
) -> None:
    """Raise unless a deployment satisfies paper-style memory and energy limits."""

    values = np.asarray(deployment, dtype=np.int64)
    system = config.system
    if values.shape != (system.num_layers,):
        raise ValueError(f"deployment must have shape ({system.num_layers},)")
    if np.any(values < 0) or np.any(values >= system.num_uavs):
        raise ValueError("deployment contains an invalid UAV index")
    usage = resource_usage(values, config, channel=channel, validate_indices=False)
    if np.any(usage.memory_units > np.asarray(config.uav_memory_capacity_units) + 1e-9):
        raise ValueError("deployment exceeds a UAV memory capacity")
    if np.any(usage.total_energy_joule > np.asarray(config.uav_energy_budget_joule) + 1e-9):
        raise ValueError("deployment exceeds a UAV energy budget")


def resource_usage(
    deployment: np.ndarray,
    config: ResourceConstrainedConfig,
    *,
    channel: np.ndarray | None = None,
    validate_indices: bool = True,
) -> ResourceUsage:
    """Compute memory, computation energy, and optional channel-dependent link energy."""

    values = np.asarray(deployment, dtype=np.int64)
    system = config.system
    if validate_indices:
        if values.shape != (system.num_layers,):
            raise ValueError(f"deployment must have shape ({system.num_layers},)")
        if np.any(values < 0) or np.any(values >= system.num_uavs):
            raise ValueError("deployment contains an invalid UAV index")
    memory = np.bincount(
        values,
        weights=np.asarray(config.layer_memory_units, dtype=np.float64),
        minlength=system.num_uavs,
    )
    compute_seconds = np.asarray(config.layer_compute_seconds_at_unit_speed, dtype=np.float64)
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    computation_energy = np.bincount(
        values,
        weights=config.compute_energy_coefficient * speeds[values] ** 2 * compute_seconds,
        minlength=system.num_uavs,
    )
    communication_energy = np.zeros(system.num_uavs, dtype=np.float64)
    if channel is not None:
        gains = np.asarray(channel, dtype=np.float64)
        if gains.shape != (system.num_uavs, system.num_uavs):
            raise ValueError("channel has the wrong shape")
        activations = np.asarray(config.activation_mbit_by_boundary, dtype=np.float64)
        boundaries = np.flatnonzero(values[:-1] != values[1:])
        coefficients: list[float] = []
        senders: list[int] = []
        for boundary in boundaries:
            sender = int(values[boundary])
            receiver = int(values[boundary + 1])
            spectral_efficiency = np.log2(
                1.0 + system.transmit_power * gains[sender, receiver] / system.noise_power
            )
            coefficients.append(float(activations[boundary] / spectral_efficiency))
            senders.append(sender)
        if coefficients:
            roots = np.sqrt(coefficients)
            bandwidth = system.total_bandwidth_mhz * roots / roots.sum()
            for sender, coefficient, allocated_bandwidth in zip(
                senders, coefficients, bandwidth, strict=True
            ):
                communication_energy[sender] += system.transmit_power * coefficient / allocated_bandwidth
    total = computation_energy + communication_energy + np.asarray(
        config.uav_hover_energy_joule, dtype=np.float64
    )
    return ResourceUsage(
        memory_units=memory,
        computation_energy_joule=computation_energy,
        communication_energy_joule=communication_energy,
        total_energy_joule=total,
    )


def layerwise_drop_probabilities(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
) -> np.ndarray:
    """Return a drop probability for every layer boundary without continuity assumptions."""

    validate_layerwise_deployment(deployment, config, channel=channel)
    values = np.asarray(deployment, dtype=np.int64)
    probabilities = np.zeros(config.system.num_layers - 1, dtype=np.float32)
    for boundary in np.flatnonzero(values[:-1] != values[1:]):
        probabilities[boundary] = packet_drop_probability(
            float(channel[values[boundary], values[boundary + 1]]), config.system
        )
    return probabilities


def layerwise_latency(
    deployment: np.ndarray,
    channel: np.ndarray,
    config: ResourceConstrainedConfig,
) -> ResourceLatency:
    """Evaluate paper-style computation plus optimal shared-bandwidth latency."""

    validate_layerwise_deployment(deployment, config, channel=channel)
    values = np.asarray(deployment, dtype=np.int64)
    system = config.system
    speeds = np.asarray(system.compute_speed, dtype=np.float64)
    computation = float(
        np.sum(np.asarray(config.layer_compute_seconds_at_unit_speed) / speeds[values])
    )
    coefficients: list[float] = []
    activation = np.asarray(config.activation_mbit_by_boundary, dtype=np.float64)
    for boundary in np.flatnonzero(values[:-1] != values[1:]):
        gain = float(channel[values[boundary], values[boundary + 1]])
        spectral_efficiency = np.log2(
            1.0 + system.transmit_power * gain / system.noise_power
        )
        coefficients.append(float(activation[boundary] / spectral_efficiency))
    communication = 0.0
    if coefficients:
        communication = float(np.square(np.sqrt(coefficients).sum()) / system.total_bandwidth_mhz)
    return ResourceLatency(
        computation_seconds=computation,
        communication_seconds=communication,
        total_seconds=computation + communication,
    )
