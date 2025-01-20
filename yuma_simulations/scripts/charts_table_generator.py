from dataclasses import replace
from yuma_simulations.yumas.utils import (
    YumaParams,
    SimulationHyperparameters,
    YumaSimulationNames,
)
from yuma_simulations.simulation_utils import (
    generate_chart_table,
)
from yuma_simulations.cases import cases


def main():
    # Setting global simulation parameters
    simulation_hyperparameters = SimulationHyperparameters(
        bond_penalty=1.0,
    )
    # Make sure the output file name matches the bond_penalty parameter
    file_name = "simulation_results_b1.0.html"

    # Setting individual yuma simulations parameters
    base_yuma_params = YumaParams()
    liquid_alpha_on_yuma_params = YumaParams(
        liquid_alpha=True,
    )
    yuma4_params = YumaParams(
        bond_alpha=0.025,
        alpha_high=0.99,
        alpha_low=0.9,
    )
    yuma4_liquid_params = replace(yuma4_params, liquid_alpha=True)

    yumas = YumaSimulationNames()
    yuma_versions = [
        (yumas.YUMA_RUST, base_yuma_params),
        (yumas.YUMA, base_yuma_params),
        (yumas.YUMA_LIQUID, liquid_alpha_on_yuma_params),
        (yumas.YUMA2, base_yuma_params),
        (yumas.YUMA3, base_yuma_params),
        (yumas.YUMA31, base_yuma_params),
        (yumas.YUMA32, base_yuma_params),
        (yumas.YUMA3B, base_yuma_params),
        (yumas.YUMA3B1, base_yuma_params),
        (yumas.YUMA3B2, base_yuma_params),
        (yumas.YUMA4, yuma4_params),
        (yumas.YUMA4_LIQUID, yuma4_liquid_params),
        (yumas.YUMA4B, yuma4_params),
        (yumas.YUMA4B_LIQUID, yuma4_liquid_params),
    ]

    chart_table = generate_chart_table(
        cases, yuma_versions, simulation_hyperparameters, draggable_table=True
    )

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(chart_table.data)

    print(f"HTML saved to {file_name}")


if __name__ == "__main__":
    main()
