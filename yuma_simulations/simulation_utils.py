"""
This module provides functionalities to run Yuma simulations, generate charts, and produce tables of results.
It integrates various Yuma versions, handles different chart types, and organizes the outputs into HTML tables.
"""

import pandas as pd
import torch
from IPython.display import HTML
from yuma_simulations.cases import BaseCase
from yuma_simulations.yumas import (
    Yuma,
    Yuma2,
    Yuma3,
    Yuma4,
    YumaRust,
    YumaSimulationNames,
    YumaParams,
    YumaConfig,
    SimulationHyperparameters,
)
from yuma_simulations.charts_utils import (
    _plot_validator_server_weights,
    _plot_dividends,
    _plot_bonds,
    _plot_incentives,
    _calculate_total_dividends,
)


def run_simulation(
    case: BaseCase,
    yuma_version: str,
    yuma_config: YumaConfig,
) -> tuple[dict[str, list[float]], list[torch.Tensor], list[torch.Tensor]]:
    """Runs the Yuma simulation for a given case and Yuma version, returning dividends, bonds and incentive data."""

    dividends_per_validator: dict[str, list[float]] = {
        validator: [] for validator in case.validators
    }
    bonds_per_epoch: list[torch.Tensor] = []
    server_incentives_per_epoch: list[torch.Tensor] = []
    B_state: torch.Tensor | None = None
    W_prev: torch.Tensor | None = None
    server_consensus_weight: torch.Tensor | None = None

    simulation_names = YumaSimulationNames()

    for epoch in range(case.num_epochs):
        W: torch.Tensor = case.weights_epochs[epoch]
        S: torch.Tensor = case.stakes_epochs[epoch]

        stakes_tao: torch.Tensor = S * yuma_config.total_subnet_stake
        stakes_units: torch.Tensor = stakes_tao / 1000.0

        # Call the appropriate Yuma function
        if yuma_version in [simulation_names.YUMA, simulation_names.YUMA_LIQUID]:
            result = Yuma(W=W, S=S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
        elif yuma_version == simulation_names.YUMA2:
            result = Yuma2(W=W, W_prev=W_prev, S=S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
            W_prev = result["weight"]
        elif yuma_version == simulation_names.YUMA3:
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
        elif yuma_version == simulation_names.YUMA31:
            if B_state is not None and epoch == case.reset_bonds_epoch:
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
        elif yuma_version == simulation_names.YUMA32:
            if (
                B_state is not None
                and epoch == case.reset_bonds_epoch
                and server_consensus_weight is not None
                and server_consensus_weight[case.reset_bonds_index] == 0.0
            ):
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
            server_consensus_weight = result["server_consensus_weight"]
        elif yuma_version in [simulation_names.YUMA4, simulation_names.YUMA4_LIQUID]:
            if (
                B_state is not None
                and epoch == case.reset_bonds_epoch
                and server_consensus_weight is not None
                and server_consensus_weight[case.reset_bonds_index] == 0.0
            ):
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma4(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_bonds"]
            server_consensus_weight = result["server_consensus_weight"]
        elif yuma_version == "Yuma 0 (subtensor)":
            result = YumaRust(W, S, B_old=B_state, config=yuma_config)
            B_state = result["validator_ema_bond"]
        else:
            raise ValueError("Invalid Yuma function.")

        D_normalized: torch.Tensor = result["validator_reward_normalized"]

        E_i: torch.Tensor = yuma_config.validator_emission_ratio * D_normalized
        validator_emission: torch.Tensor = E_i * yuma_config.total_epoch_emission

        for i, validator in enumerate(case.validators):
            stake_unit = float(stakes_units[i].item())
            validator_emission_i = float(validator_emission[i].item())
            if stake_unit > 1e-6:
                dividend_per_1000_tao = validator_emission_i / stake_unit
            else:
                dividend_per_1000_tao = 0.0
            dividends_per_validator[validator].append(dividend_per_1000_tao)

        bonds_per_epoch.append(B_state.clone())
        server_incentives_per_epoch.append(result["server_incentive"])

    return dividends_per_validator, bonds_per_epoch, server_incentives_per_epoch


def generate_chart_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    yuma_hyperparameters: SimulationHyperparameters,
    draggable_table: bool = False,
) -> HTML:
    """Generates an HTML table with embedded charts for given cases and Yuma versions."""
    table_data: dict[str, list[str]] = {
        yuma_version: [] for yuma_version, _ in yuma_versions
    }

    def process_chart(
        table_data: dict[str, list[str]], chart_base64_dict: dict[str, str]
    ) -> None:
        for yuma_version, chart_base64 in chart_base64_dict.items():
            content = f"{chart_base64}"
            table_data[yuma_version].append(content)

    for idx, case in enumerate(cases):
        if idx in [9, 10]:
            chart_types = [
                "weights",
                "dividends",
                "bonds",
                "normalized_bonds",
                "incentives",
            ]
        else:
            chart_types = ["weights", "dividends", "bonds", "normalized_bonds"]

        for chart_type in chart_types:
            chart_base64_dict: dict[str, str] = {}
            for yuma_version, yuma_params in yuma_versions:
                yuma_config = YumaConfig(
                    simulation=yuma_hyperparameters,
                    yuma_params=yuma_params,
                )
                yuma_names = YumaSimulationNames()
                full_case_name = f"{case.name} - {yuma_version}"
                if yuma_version in [
                    yuma_names.YUMA,
                    yuma_names.YUMA_LIQUID,
                    yuma_names.YUMA2,
                ]:
                    full_case_name = (
                        f"{full_case_name} - beta={yuma_config.bond_penalty}"
                    )
                elif yuma_version in [yuma_names.YUMA4_LIQUID]:
                    full_case_name = f"{full_case_name} [{yuma_config.alpha_low}, {yuma_config.alpha_high}]"

                (
                    dividends_per_validator,
                    bonds_per_epoch,
                    server_incentives_per_epoch,
                ) = run_simulation(
                    case=case,
                    yuma_version=yuma_version,
                    yuma_config=yuma_config,
                )

                if chart_type == "weights":
                    chart_base64 = _plot_validator_server_weights(
                        validators=case.validators,
                        weights_epochs=case.weights_epochs,
                        servers=case.servers,
                        num_epochs=case.num_epochs,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                elif chart_type == "dividends":
                    chart_base64 = _plot_dividends(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        dividends_per_validator=dividends_per_validator,
                        case=full_case_name,
                        base_validator=case.base_validator,
                        to_base64=True,
                    )
                elif chart_type == "bonds":
                    chart_base64 = _plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                elif chart_type == "normalized_bonds":
                    chart_base64 = _plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True,
                        normalize=True,
                    )
                elif chart_type == "incentives":
                    chart_base64 = _plot_incentives(
                        servers=case.servers,
                        server_incentives_per_epoch=server_incentives_per_epoch,
                        num_epochs=case.num_epochs,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                else:
                    raise ValueError("Invalid chart type.")

                chart_base64_dict[yuma_version] = chart_base64

            process_chart(table_data, chart_base64_dict)

    summary_table = pd.DataFrame(table_data)

    if draggable_table:
        full_html = _generate_draggable_html_table(table_data, summary_table)
    else:
        full_html = _generate_ipynb_table(table_data, summary_table)

    return HTML(full_html)


def _generate_draggable_html_table(
    table_data: dict[str, list[str]], summary_table: pd.DataFrame
) -> str:
    custom_css_js = """
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        
        .scrollable-table-container {
            background-color: #FFFFFF; 
            width: 100%; 
            height: 100vh;
            overflow: auto;
            border: 1px solid #ccc;
            position: relative; 
            user-select: none;
            scrollbar-width: auto;
            -ms-overflow-style: auto;
            cursor: grab;
        }

        .scrollable-table-container:active {
            cursor: grabbing;
        }

        /* Ensure images don't intercept pointer events */
        .scrollable-table-container img {
            user-select: none;
            -webkit-user-drag: none;
            pointer-events: none; /* This ensures the container gets the events */
        }
        
        .scrollable-table-container::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        table {
            border-collapse: collapse;
            margin: 0;
            width: auto;
        }
        
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }

        tbody tr:nth-child(odd) {
            background-color: #FFFFFF;
        }
        tbody tr:nth-child(even) {
            background-color: #F8F8F8;
        }
    </style>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            const container = document.querySelector('.scrollable-table-container');
            let isDown = false;
            let startX, startY, scrollLeft, scrollTop;
            
            container.addEventListener('dragstart', function(e) {
                e.preventDefault();
            });

            container.addEventListener('mousedown', (e) => {
                e.preventDefault();
                isDown = true;
                startX = e.clientX;
                startY = e.clientY;
                scrollLeft = container.scrollLeft;
                scrollTop = container.scrollTop;
            });

            document.addEventListener('mouseup', () => {
                isDown = false;
            });

            document.addEventListener('mousemove', (e) => {
                if(!isDown) return;
                e.preventDefault();
                const x = e.clientX;
                const y = e.clientY;
                const walkX = x - startX;
                const walkY = y - startY;
                container.scrollLeft = scrollLeft - walkX;
                container.scrollTop = scrollTop - walkY;
            });
        });
    </script>
    """

    # Generate HTML rows
    html_rows: list[str] = []
    for i in range(len(next(iter(table_data.values())))):
        row_html = "<tr>"
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{''.join(f'<th>{col}</th>' for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """

    return custom_css_js + html_table


def _generate_ipynb_table(
    table_data: dict[str, list[str]], summary_table: pd.DataFrame
) -> str:
    custom_css = """
    <style>
        .scrollable-table-container {
            background-color: #FFFFFF; /* Ensure container is white */
            width: 100%; 
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            border: 1px solid #ccc;
        }
        table {
            border-collapse: collapse;
            table-layout: auto;
            width: auto;
        }
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }
        /* Add alternating row colors */
        tbody tr:nth-child(odd) {
            background-color: #FFFFFF; /* White for odd rows */
        }
        tbody tr:nth-child(even) {
            background-color: #F8F8F8; /* Light gray for even rows */
        }
    </style>
    """

    html_rows: list[str] = []
    num_rows = len(next(iter(table_data.values())))
    for i in range(num_rows):
        row_html = "<tr>"
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f"<td>{cell_content}</td>"
        row_html += "</tr>"
        html_rows.append(row_html)

    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{''.join(f'<th>{col}</th>' for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    return custom_css + html_table


def generate_total_dividends_table(
    cases: list[BaseCase],
    yuma_versions: list[tuple[str, YumaParams]],
    simulation_hyperparameters: SimulationHyperparameters,
) -> pd.DataFrame:
    """Generates a DataFrame of total dividends for standardized validator names across Yuma versions."""

    standardized_validators = ["Validator A", "Validator B", "Validator C"]
    rows: list[dict[str, object]] = []

    for case in cases:
        if len(case.validators) != 3:
            raise ValueError(f"Case '{case.name}' does not have exactly 3 validators.")

        validator_mapping = {
            case.validators[0]: "Validator A",
            case.validators[1]: "Validator B",
            case.validators[2]: "Validator C",
        }

        row: dict[str, object] = {"Case": case.name}

        for yuma_version, yuma_params in yuma_versions:
            yuma_config = YumaConfig(
                simulation=simulation_hyperparameters,
                yuma_params=yuma_params,
            )

            dividends_per_validator, _, _ = run_simulation(
                case=case,
                yuma_version=yuma_version,
                yuma_config=yuma_config,
            )

            total_dividends, _ = _calculate_total_dividends(
                validators=case.validators,
                dividends_per_validator=dividends_per_validator,
                base_validator=case.base_validator,
                num_epochs=case.num_epochs,
            )

            standardized_dividends = {
                validator_mapping[orig_val]: total_dividends.get(orig_val, 0.0)
                for orig_val in case.validators
            }

            for std_validator in standardized_validators:
                dividend = standardized_dividends.get(std_validator, 0.0)
                column_name = f"{std_validator} - {yuma_version}"
                row[column_name] = dividend

        rows.append(row)

    df = pd.DataFrame(rows)
    columns = ["Case"]
    for yuma_version, _ in yuma_versions:
        for std_validator in standardized_validators:
            col_name = f"{std_validator} - {yuma_version}"
            if col_name in df.columns:
                columns.append(col_name)
    df = df[columns]

    return df
