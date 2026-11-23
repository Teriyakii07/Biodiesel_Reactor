
"""
BIODIESEL REACTOR - NUMERICAL ANALYSIS
50-Liter Soybean Oil Transesterification Reactor

Features:
1. Newton-Raphson root finding
2. Heating and cooling energy estimation
3. Sensitivity analysis
4. Operating condition optimization
5. Root finding at optimized conditions
"""

import numpy as np
import pandas as pd

from dataclasses import replace

from reactor import ReactorParameters, run_simulation


# ==========================================================
# 1. DEFAULT ANALYSIS SETTINGS
# ==========================================================

TARGET_CONVERSION = 96.5

# Illustrative electricity rate.
# Change this based on your actual electricity tariff.
ELECTRICITY_RATE = 10.00  # PHP/kWh

# Assumed utility equipment efficiencies.
HEATER_EFFICIENCY = 0.90
COOLING_COP = 3.0

# Sensitivity analysis ranges.
METHANOL_RATIOS = np.linspace(3, 12, 10)
TEMPERATURES = np.linspace(45, 65, 9)


# ==========================================================
# 2. NUMERICAL INTEGRATION
# ==========================================================

def trapezoidal_integral(y, x):
    """
    Calculate an integral using the trapezoidal rule.
    Custom implementation for NumPy compatibility.
    """

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if len(y) != len(x):
        raise ValueError(
            "Arrays must have equal lengths."
        )

    if len(x) < 2:
        return 0.0

    return np.sum(
        0.5 * (y[:-1] + y[1:]) * np.diff(x)
    )


# ==========================================================
# 3. FAME CONVERSION CALCULATION
# ==========================================================

def calculate_conversion(params, holding_time):
    """
    Calculate FAME conversion at a specified batch time.
    """

    if holding_time < 0:
        raise ValueError(
            "Holding time cannot be negative."
        )

    # At zero minutes, no reaction has occurred.
    # Do not call run_simulation with final_time = 0.
    if holding_time == 0:
        return 0.0

    test_params = replace(
        params,
        final_time=float(holding_time)
    )

    results = run_simulation(test_params)

    return float(
        results["FAME Conversion (%)"].iloc[-1]
    )


# ==========================================================
# 4. NEWTON-RAPHSON ROOT FINDING
# ==========================================================

def newton_raphson(
    params,
    target=TARGET_CONVERSION,
    initial_guess=90.0,
    tolerance=1e-5,
    time_tolerance=1e-5,
    max_iterations=40
):
    """
    Find the holding time required to reach the target
    FAME conversion using a safeguarded Newton-Raphson method.

    The derivative is estimated using finite differences.

    If a Newton step is invalid or leaves the valid interval,
    the method uses a bisection step instead.
    """

    if not 0 < target <= 100:
        raise ValueError(
            "Target conversion must be between 0 and 100."
        )

    if params.final_time <= 0:
        raise ValueError(
            "Final simulation time must be positive."
        )

    if not 0 <= initial_guess <= params.final_time:
        raise ValueError(
            "Initial guess must be within the batch duration."
        )

    # ------------------------------------------------------
    # Check conversion at time zero.
    # ------------------------------------------------------

    initial_conversion = calculate_conversion(
        params,
        0.0
    )

    if initial_conversion >= target:
        return {
            "converged": True,
            "holding_time": 0.0,
            "conversion": initial_conversion,
            "iterations": 0,
            "method": "Newton-Raphson",
            "message": "Target conversion is already reached."
        }

    # ------------------------------------------------------
    # Check conversion at the end of the batch.
    # ------------------------------------------------------

    final_conversion = calculate_conversion(
        params,
        params.final_time
    )

    if final_conversion < target:
        return {
            "converged": False,
            "holding_time": None,
            "conversion": final_conversion,
            "iterations": 0,
            "method": "Newton-Raphson",
            "message": (
                f"Target of {target:.2f}% cannot be reached "
                f"within {params.final_time:.2f} minutes. "
                f"Final simulated conversion: "
                f"{final_conversion:.4f}%."
            )
        }

    # ------------------------------------------------------
    # Establish the initial bracket.
    # ------------------------------------------------------

    lower_time = 0.0
    upper_time = float(params.final_time)

    t = float(initial_guess)

    # Ensure the initial estimate is not exactly at
    # either endpoint of the bracket.
    if t <= lower_time or t >= upper_time:
        t = (lower_time + upper_time) / 2

    # ------------------------------------------------------
    # Newton-Raphson iterations.
    # ------------------------------------------------------

    for iteration in range(1, max_iterations + 1):

        current_conversion = calculate_conversion(
            params,
            t
        )

        error = current_conversion - target

        # Check conversion convergence.
        if abs(error) <= tolerance:
            return {
                "converged": True,
                "holding_time": t,
                "conversion": current_conversion,
                "iterations": iteration,
                "method": "Safeguarded Newton-Raphson",
                "message": (
                    "Newton-Raphson converged successfully."
                )
            }

        # Update the bracket.
        if error < 0:
            lower_time = t
        else:
            upper_time = t

        # Check time interval convergence.
        if upper_time - lower_time <= time_tolerance:

            midpoint = (
                lower_time + upper_time
            ) / 2

            midpoint_conversion = calculate_conversion(
                params,
                midpoint
            )

            return {
                "converged": (
                    abs(midpoint_conversion - target)
                    <= max(tolerance, 1e-4)
                ),
                "holding_time": midpoint,
                "conversion": midpoint_conversion,
                "iterations": iteration,
                "method": "Safeguarded Newton-Raphson",
                "message": (
                    "Time interval reached the specified tolerance."
                )
            }

        # --------------------------------------------------
        # Estimate derivative using finite differences.
        # --------------------------------------------------

        delta = min(
            0.01,
            max((upper_time - lower_time) * 0.1, 1e-6)
        )

        left_time = max(
            lower_time,
            t - delta
        )

        right_time = min(
            upper_time,
            t + delta
        )

        if right_time == left_time:
            new_t = (
                lower_time + upper_time
            ) / 2

        else:

            left_conversion = calculate_conversion(
                params,
                left_time
            )

            right_conversion = calculate_conversion(
                params,
                right_time
            )

            derivative = (
                right_conversion - left_conversion
            ) / (right_time - left_time)

            # ----------------------------------------------
            # Newton-Raphson update.
            # ----------------------------------------------

            if abs(derivative) > 1e-10:

                new_t = t - error / derivative

            else:

                new_t = np.nan

            # If the Newton step is invalid or outside
            # the bracket, use the bisection midpoint.
            if (
                not np.isfinite(new_t)
                or new_t <= lower_time
                or new_t >= upper_time
            ):

                new_t = (
                    lower_time + upper_time
                ) / 2

        t = new_t

    # ------------------------------------------------------
    # Maximum iterations reached.
    # ------------------------------------------------------

    final_estimate = calculate_conversion(
        params,
        t
    )

    return {
        "converged": False,
        "holding_time": t,
        "conversion": final_estimate,
        "iterations": max_iterations,
        "method": "Safeguarded Newton-Raphson",
        "message": (
            "Maximum Newton-Raphson iterations reached."
        )
    }


# ==========================================================
# 5. HEATING AND COOLING ENERGY COST
# ==========================================================

def calculate_energy_cost(
    data,
    params,
    electricity_rate=ELECTRICITY_RATE,
    efficiency=HEATER_EFFICIENCY,
    cooling_cop=COOLING_COP
):
    """
    Estimate heating and cooling utility electricity.

    Positive jacket duty = heating.
    Negative jacket duty = cooling.

    Energy is calculated using the reactor jacket heat duty.
    """

    if electricity_rate < 0:
        raise ValueError(
            "Electricity rate cannot be negative."
        )

    if not 0 < efficiency <= 1:
        raise ValueError(
            "Heater efficiency must be between 0 and 1."
        )

    if cooling_cop <= 0:
        raise ValueError(
            "Cooling COP must be greater than zero."
        )

    time = data[
        "Time (min)"
    ].to_numpy(dtype=float)

    temperature = data[
        "Temperature (C)"
    ].to_numpy(dtype=float)

    jacket_temperature = params.jacket_temperature

    # Jacket heat transfer in J/min.
    jacket_duty = (
        params.UA
        * 60.0
        * (jacket_temperature - temperature)
    )

    # Separate heating and cooling duties.
    heating_duty = np.maximum(
        jacket_duty,
        0
    )

    cooling_duty = np.maximum(
        -jacket_duty,
        0
    )

    # Integrate energy using the custom trapezoidal rule.
    heating_energy_j = trapezoidal_integral(
        heating_duty,
        time
    )

    cooling_energy_j = trapezoidal_integral(
        cooling_duty,
        time
    )

    # Convert Joules to kWh.
    heating_energy_kwh = (
        heating_energy_j / 3_600_000
    )

    cooling_energy_kwh = (
        cooling_energy_j / 3_600_000
    )

    # Estimate electrical energy consumption.
    electrical_heating_kwh = (
        heating_energy_kwh / efficiency
    )

    electrical_cooling_kwh = (
        cooling_energy_kwh / cooling_cop
    )

    total_electrical_energy = (
        electrical_heating_kwh
        + electrical_cooling_kwh
    )

    # Calculate electricity cost.
    energy_cost = (
        total_electrical_energy
        * electricity_rate
    )

    return {
        "heating_energy_kWh": heating_energy_kwh,

        "cooling_energy_kWh": cooling_energy_kwh,

        "electrical_heating_kWh": (
            electrical_heating_kwh
        ),

        "electrical_cooling_kWh": (
            electrical_cooling_kwh
        ),

        "electrical_energy_kWh": (
            total_electrical_energy
        ),

        "energy_cost_PHP": energy_cost,

        "electricity_rate_PHP_kWh": electricity_rate
    }


# ==========================================================
# 6. SENSITIVITY ANALYSIS
# ==========================================================

def sensitivity_analysis(
    base_params,
    ratios=None,
    temperatures=None,
    electricity_rate=ELECTRICITY_RATE
):
    """
    Evaluate the effects of methanol-to-oil ratio
    and initial reactor temperature.

    Outputs:
    - Final FAME yield
    - Maximum reactor temperature
    - Heating energy
    - Cooling energy
    - Estimated electrical energy
    - Estimated energy cost
    - Safety warning
    """

    if ratios is None:
        ratios = METHANOL_RATIOS

    if temperatures is None:
        temperatures = TEMPERATURES

    records = []

    for ratio in ratios:

        for temperature in temperatures:

            test_params = replace(
                base_params,
                methanol_ratio=float(ratio),
                initial_temperature=float(temperature)
            )

            # Run reactor simulation.
            data = run_simulation(test_params)

            # Get final FAME conversion.
            final_conversion = float(
                data["FAME Conversion (%)"].iloc[-1]
            )

            # Get maximum reactor temperature.
            maximum_temperature = float(
                data["Temperature (C)"].max()
            )

            # Calculate energy consumption and cost.
            energy = calculate_energy_cost(
                data,
                test_params,
                electricity_rate=electricity_rate
            )

            # Store simulation results.
            records.append({

                "Methanol Ratio": float(ratio),

                "Initial Temperature (C)": float(
                    temperature
                ),

                "Final FAME Yield (%)": (
                    final_conversion
                ),

                "Maximum Temperature (C)": (
                    maximum_temperature
                ),

                "Heating Energy (kWh)": (
                    energy["heating_energy_kWh"]
                ),

                "Cooling Energy (kWh)": (
                    energy["cooling_energy_kWh"]
                ),

                "Electrical Energy (kWh)": (
                    energy["electrical_energy_kWh"]
                ),

                "Net Batch Energy Cost (PHP)": (
                    energy["energy_cost_PHP"]
                ),

                "Safety Warning": (
                    maximum_temperature >= 64.7
                )

            })

    return pd.DataFrame(records)


# ==========================================================
# 7. OPERATING CONDITION OPTIMIZATION
# ==========================================================

def find_optimal_conditions(
    sensitivity_data,
    minimum_yield=TARGET_CONVERSION
):
    """
    Find the lowest-energy-cost operating condition
    that meets the minimum FAME yield.

    Conditions reaching or exceeding 64.7 C
    are excluded from safe operating candidates.

    If no safe condition meets the yield target,
    return the highest-yield safe condition.
    """

    if sensitivity_data.empty:
        raise ValueError(
            "Sensitivity analysis data is empty."
        )

    # Filter conditions meeting yield and temperature limits.
    valid_conditions = sensitivity_data[
        (
            sensitivity_data["Final FAME Yield (%)"]
            >= minimum_yield
        )
        &
        (
            ~sensitivity_data["Safety Warning"]
        )
    ]

    if not valid_conditions.empty:

        optimal = valid_conditions.loc[
            valid_conditions[
                "Net Batch Energy Cost (PHP)"
            ].idxmin()
        ]

        status = (
            "Meets the FAME yield target and "
            "temperature safety constraint."
        )

    else:

        # Find the highest-yield safe condition.
        safe_conditions = sensitivity_data[
            ~sensitivity_data["Safety Warning"]
        ]

        if safe_conditions.empty:

            candidates = sensitivity_data

            status = (
                "No tested condition is below the "
                "temperature safety threshold."
            )

        else:

            candidates = safe_conditions

            status = (
                "No tested safe condition reaches the "
                "required yield. Showing the highest-yield "
                "safe operating condition."
            )

        optimal = candidates.loc[
            candidates[
                "Final FAME Yield (%)"
            ].idxmax()
        ]

    return {

        "methanol_ratio": float(
            optimal["Methanol Ratio"]
        ),

        "initial_temperature": float(
            optimal["Initial Temperature (C)"]
        ),

        "fame_yield": float(
            optimal["Final FAME Yield (%)"]
        ),

        "energy_cost": float(
            optimal["Net Batch Energy Cost (PHP)"]
        ),

        "maximum_temperature": float(
            optimal["Maximum Temperature (C)"]
        ),

        "status": status

    }


# ==========================================================
# 8. RUN COMPLETE ANALYSIS
# ==========================================================

def run_complete_analysis(params):
    """
    Run root finding, sensitivity analysis,
    and operating condition optimization.
    """

    # 1. Newton-Raphson root finding.
    root_result = newton_raphson(params)

    # 2. Sensitivity analysis.
    sensitivity_data = sensitivity_analysis(params)

    # 3. Operating condition optimization.
    optimization_result = find_optimal_conditions(
        sensitivity_data
    )

    return {

        "root_finding": root_result,

        "sensitivity": sensitivity_data,

        "optimization": optimization_result

    }


# ==========================================================
# 9. TEST MODULE
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BIODIESEL REACTOR - ANALYSIS MODULE")
    print("=" * 60)

    # Create default reactor parameters.
    params = ReactorParameters()

    # ------------------------------------------------------
    # 1. DEFAULT NEWTON-RAPHSON
    # ------------------------------------------------------

    print("\n1. NEWTON-RAPHSON ROOT FINDING")
    print("-" * 60)

    root_result = newton_raphson(params)

    for key, value in root_result.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------
    # 2. SENSITIVITY ANALYSIS
    # ------------------------------------------------------

    print("\n2. SENSITIVITY ANALYSIS")
    print("-" * 60)

    sensitivity_data = sensitivity_analysis(
        params,
        ratios=[3, 6, 9, 12],
        temperatures=[45, 55, 65]
    )

    print(
        sensitivity_data.to_string(
            index=False,
            col_space=3,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print(
        "\nTotal simulation cases:",
        len(sensitivity_data)
    )

    # ------------------------------------------------------
    # 3. OPERATING CONDITION OPTIMIZATION
    # ------------------------------------------------------

    print("\n3. OPERATING CONDITION OPTIMIZATION")
    print("-" * 60)

    optimization_result = find_optimal_conditions(
        sensitivity_data
    )

    for key, value in optimization_result.items():
        print(f"{key}: {value}")

    # ------------------------------------------------------
    # 4. ROOT FINDING AT OPTIMIZED CONDITIONS
    # ------------------------------------------------------

    print("\n4. ROOT FINDING AT OPTIMIZED CONDITIONS")
    print("-" * 60)

    optimized_params = replace(
        params,
        methanol_ratio=optimization_result["methanol_ratio"],
        initial_temperature=optimization_result[
            "initial_temperature"
        ]
    )

    optimized_root = newton_raphson(
        optimized_params,
        target=TARGET_CONVERSION,
        initial_guess=110.0
    )

    for key, value in optimized_root.items():
        print(f"{key}: {value}")

    print("\nAnalysis module completed.")
