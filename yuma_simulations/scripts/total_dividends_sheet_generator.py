from yuma_simulations.cases import cases
from yuma_simulations.simulation_utils import generate_total_dividends_table
from yuma_simulations.yumas.utils import (
    YumaParams,
    SimulationHyperparameters,
    YumaSimulationNames,
)


def main():
    # Define simulation hyperparameters
    simulation_hyperparameters = SimulationHyperparameters(
        bond_penalty=0.0,
    )
    # Make sure the output file name matches the bond_penalty parameter
    file_name = "total_dividends_b0.csv"

    # Define Yuma parameter variations
    base_yuma_params = YumaParams()
    liquid_alpha_on_yuma_params = YumaParams(
        liquid_alpha=True,
    )

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
        (yumas.YUMA4, base_yuma_params),
        (yumas.YUMA4_LIQUID, liquid_alpha_on_yuma_params),
        (yumas.YUMA4B, yuma4_params),
        (yumas.YUMA4B_LIQUID, yuma4_liquid_params),
    ]

    print("Starting generation of total dividends table.")
    dividends_df = generate_total_dividends_table(
        cases=cases,
        yuma_versions=yuma_versions,
        simulation_hyperparameters=simulation_hyperparameters,
    )

    # Check for missing values
    if dividends_df.isnull().values.any():
        print("CSV contains missing values. Please check the simulation data.")
    else:
        print("No missing values detected in the CSV data.")

    # Save the DataFrame to a CSV file
    dividends_df.to_csv(file_name, index=False, float_format="%.6f")
    print(f"CSV file {file_name} has been created successfully.")


if __name__ == "__main__":
    main()
