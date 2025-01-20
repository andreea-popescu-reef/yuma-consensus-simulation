from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class SimulationHyperparameters:
    kappa: float = 0.5
    bond_penalty: float = 1.0
    total_epoch_emission: float = 100.0
    validator_emission_ratio: float = 0.41
    total_subnet_stake: float = 1_000_000.0
    consensus_precision: int = 100_000


@dataclass
class YumaParams:
    bond_alpha: float = 0.1
    liquid_alpha: bool = False
    alpha_high: float = 0.9
    alpha_low: float = 0.7
    decay_rate: float = 0.1
    capacity_alpha: float = 0.1
    override_consensus_high: Optional[float] = None
    override_consensus_low: Optional[float] = None


@dataclass
class YumaConfig:
    simulation: SimulationHyperparameters = field(
        default_factory=SimulationHyperparameters
    )
    yuma_params: YumaParams = field(default_factory=YumaParams)

    def __post_init__(self):
        # Flatten fields for direct access
        simulation_dict = asdict(self.simulation)
        yuma_params_dict = asdict(self.yuma_params)

        for key, value in simulation_dict.items():
            setattr(self, key, value)

        for key, value in yuma_params_dict.items():
            setattr(self, key, value)


@dataclass(frozen=True)
class YumaSimulationNames:
    YUMA_RUST: str = "Yuma 0 (subtensor)"
    YUMA: str = "Yuma 1 (paper)"
    YUMA_LIQUID: str = "Yuma 1 (paper) - liquid alpha on"
    YUMA2: str = "Yuma 2 (Adrian-Fish)"
    YUMA3: str = "Yuma 3 (Rhef)"
    YUMA31: str = "Yuma 3.1 (Rhef+reset)"
    YUMA32: str = "Yuma 3.2 (Rhef+conditional)"
    YUMA4: str = "Yuma 4 (Rhef+relative bonds)"
    YUMA4_LIQUID: str = "Yuma 4 (Rhef+relative bonds) - liquid alpha on"
