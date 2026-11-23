
"""
BIODIESEL BATCH REACTOR SIMULATOR
Feedstock: Soybean Oil
Reactant: Methanol
Catalyst: NaOH

Kinetic reference:
Noureddini & Zhu (1997), JAOCS, 74, 1457-1463.

Concentration units: mol/L
Time units: minutes
Temperature units: Kelvin internally
Energy units: Joules
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass


# ==========================================================
# 1. PHYSICAL AND KINETIC PARAMETERS
# ==========================================================

@dataclass
class ReactorParameters:

    # Reactor specifications
    volume: float = 50.0

    # Initial operating conditions
    initial_temperature: float = 55.0
    jacket_temperature: float = 55.0
    methanol_ratio: float = 6.0
    catalyst_wt_pct: float = 0.20

    # Integration settings
    step_size: float = 0.1
    final_time: float = 120.0

    # Heat transfer
    UA: float = 150.0  # W/K

    # Reactor thermal properties (engineering estimates)
    reactor_mass: float = 100.0  # kg
    reactor_cp: float = 500.0   # J/kg/K

    # Soybean oil and methanol properties
    oil_mw: float = 885.0       # g/mol
    oil_density: float = 0.915  # kg/L

    methanol_mw: float = 32.04  # g/mol
    methanol_density: float = 0.792  # kg/L

    # Approximate molar heat capacities (J/mol/K)
    cp_TG: float = 1682.0
    cp_DG: float = 1100.0
    cp_MG: float = 750.0
    cp_M: float = 81.0
    cp_E: float = 592.0
    cp_GL: float = 221.0

    # Overall reaction enthalpy (J/mol TG)
    # Approximate soybean-oil literature value.
    # Distributed equally across the three steps.
    total_reaction_enthalpy: float = 28300.0


# ==========================================================
# 2. LITERATURE-BASED KINETIC CONSTANTS
# ==========================================================

R = 8.314462618  # J/mol/K

REFERENCE_TEMPERATURE = 323.15  # 50 degrees Celsius

# Activation energies converted from cal/mol to J/mol.
ACTIVATION_ENERGIES = np.array([
    13145,
    9932,
    19860,
    14639,
    6421,
    9588
]) * 4.184

# Rate constants at 50 degrees Celsius.
# Units: L/mol/min
REFERENCE_RATE_CONSTANTS = np.array([
    0.050,
    0.110,
    0.215,
    1.228,
    0.242,
    0.007
])

# Calculate Arrhenius pre-exponential factors.
PRE_EXPONENTIAL_FACTORS = (
    REFERENCE_RATE_CONSTANTS *
    np.exp(
        ACTIVATION_ENERGIES /
        (R * REFERENCE_TEMPERATURE)
    )
)


def arrhenius_constants(temperature):
    """
    Calculate the six temperature-dependent rate constants.

    Parameters:
        temperature: Reactor temperature in Kelvin.

    Returns:
        Six rate constants in L/mol/min.
    """

    if temperature <= 0:
        raise ValueError("Temperature must be above absolute zero.")

    return (
        PRE_EXPONENTIAL_FACTORS *
        np.exp(
            -ACTIVATION_ENERGIES /
            (R * temperature)
        )
    )


# ==========================================================
# 3. INITIAL CONCENTRATIONS
# ==========================================================

def initial_conditions(params):
    """
    Calculate initial concentrations using the selected
    methanol-to-oil molar ratio and 50-L reactor volume.

    Assumes additive initial liquid volumes.
    """

    if params.methanol_ratio < 3:
        raise ValueError(
            "Methanol-to-oil ratio must be at least 3:1."
        )

    if params.volume <= 0:
        raise ValueError("Reactor volume must be positive.")

    if params.catalyst_wt_pct <= 0:
        raise ValueError("Catalyst percentage must be positive.")

    # Convert molecular weights from g/mol to kg/mol.
    oil_mw = params.oil_mw / 1000
    methanol_mw = params.methanol_mw / 1000

    # Initial liquid volume occupied per mole of TG,
    # including the corresponding methanol quantity.
    volume_per_mole = (
        oil_mw / params.oil_density
        +
        params.methanol_ratio *
        methanol_mw / params.methanol_density
    )

    CTG0 = 1.0 / volume_per_mole
    CM0 = params.methanol_ratio * CTG0

    # State order:
    # TG, DG, MG, Methanol, FAME, Glycerol, Temperature

    return np.array([
        CTG0,
        0.0,
        0.0,
        CM0,
        0.0,
        0.0,
        params.initial_temperature + 273.15
    ])


# ==========================================================
# 4. REVERSIBLE REACTION KINETICS
# ==========================================================

def reaction_rates(state, params):
    """
    Calculate the three reversible reaction rates.

    Reaction 1:
        TG + Methanol <-> DG + FAME

    Reaction 2:
        DG + Methanol <-> MG + FAME

    Reaction 3:
        MG + Methanol <-> Glycerol + FAME
    """

    TG, DG, MG, M, E, GL, T = state

    k1, k2, k3, k4, k5, k6 = arrhenius_constants(T)

    # Catalyst scaling:
    # Assumes first-order dependence on catalyst loading.
    # Reference loading = 0.20 wt%.
    catalyst_factor = params.catalyst_wt_pct / 0.20

    k1 *= catalyst_factor
    k2 *= catalyst_factor
    k3 *= catalyst_factor
    k4 *= catalyst_factor
    k5 *= catalyst_factor
    k6 *= catalyst_factor

    # Three reversible second-order rate expressions.
    r1 = k1 * TG * M - k2 * DG * E

    r2 = k3 * DG * M - k4 * MG * E

    r3 = k5 * MG * M - k6 * GL * E

    return r1, r2, r3


# ==========================================================
# 5. COUPLED MASS AND ENERGY BALANCES
# ==========================================================

def reactor_model(state, params):
    """
    Calculate the seven coupled differential equations.

    State:
        [TG, DG, MG, M, E, GL, T]

    Returns:
        [dTG, dDG, dMG, dM, dE, dGL, dT]
    """

    TG, DG, MG, M, E, GL, T = state

    r1, r2, r3 = reaction_rates(state, params)

    # --------------------------
    # MATERIAL BALANCES
    # --------------------------

    dTG = -r1

    dDG = r1 - r2

    dMG = r2 - r3

    dM = -(r1 + r2 + r3)

    dE = r1 + r2 + r3

    dGL = r3

    # --------------------------
    # ENERGY BALANCE
    # --------------------------

    # Reaction enthalpy is positive for an endothermic
    # reaction. Therefore, heat released to the reactor
    # is negative for positive reaction rates.

    delta_H = params.total_reaction_enthalpy / 3.0

    Q_reaction = (
        -params.volume *
        delta_H *
        (r1 + r2 + r3)
    )

    # Jacket heat transfer.
    # Convert W/K into J/min/K.
    UA_min = params.UA * 60.0

    Q_jacket = UA_min * (
        params.jacket_temperature + 273.15 - T
    )

    # Total mixture heat capacity.
    # Concentrations are in mol/L and volume in L.
    mixture_heat_capacity = params.volume * (
        TG * params.cp_TG
        + DG * params.cp_DG
        + MG * params.cp_MG
        + M * params.cp_M
        + E * params.cp_E
        + GL * params.cp_GL
    )

    reactor_heat_capacity = (
        params.reactor_mass * params.reactor_cp
    )

    total_heat_capacity = (
        mixture_heat_capacity +
        reactor_heat_capacity
    )

    if total_heat_capacity <= 0:
        raise ValueError("Invalid total heat capacity.")

    dT = (
        Q_reaction + Q_jacket
    ) / total_heat_capacity

    return np.array([
        dTG,
        dDG,
        dMG,
        dM,
        dE,
        dGL,
        dT
    ])


# ==========================================================
# 6. CUSTOM FOURTH-ORDER RUNGE-KUTTA SOLVER
# ==========================================================

def rk4_step(function, t, state, h, params):
    """
    Perform one fourth-order Runge-Kutta integration step.
    """

    k1 = function(state, params)

    k2 = function(
        state + (h / 2) * k1,
        params
    )

    k3 = function(
        state + (h / 2) * k2,
        params
    )

    k4 = function(
        state + h * k3,
        params
    )

    new_state = state + (
        h / 6
    ) * (
        k1 + 2*k2 + 2*k3 + k4
    )

    return new_state


def run_simulation(params):
    """
    Run the complete 50-liter biodiesel batch reactor
    simulation using the custom RK4 algorithm.

    Returns:
        Pandas DataFrame containing simulation results.
    """

    if params.step_size <= 0:
        raise ValueError("Step size must be positive.")

    if params.final_time <= 0:
        raise ValueError("Final time must be positive.")

    if not 0 <= params.initial_temperature <= 100:
        raise ValueError("Invalid initial temperature.")

    state = initial_conditions(params)

    time = 0.0

    results = []

    initial_TG = state[0]

    while time < params.final_time:

        # Record current state.
        TG, DG, MG, M, E, GL, T = state

        # FAME conversion based on the theoretical
        # maximum of 3 moles of FAME per mole of TG.
        fame_conversion = (
            E / (3.0 * initial_TG)
        ) * 100.0

        results.append({
            "Time (min)": time,
            "TG (mol/L)": TG,
            "DG (mol/L)": DG,
            "MG (mol/L)": MG,
            "Methanol (mol/L)": M,
            "FAME (mol/L)": E,
            "Glycerol (mol/L)": GL,
            "Temperature (C)": T - 273.15,
            "FAME Conversion (%)": fame_conversion
        })

        # Use a shorter final step when necessary.
        h = min(
            params.step_size,
            params.final_time - time
        )

        state = rk4_step(
            reactor_model,
            time,
            state,
            h,
            params
        )

        # Detect numerical instability instead of
        # silently accepting physically invalid results.
        if not np.all(np.isfinite(state)):
            raise ValueError(
                "Numerical instability detected. "
                "Try a smaller integration step."
            )

        if np.min(state[:6]) < -1e-7:
            raise ValueError(
                "Negative concentration detected. "
                "Try a smaller integration step."
            )

        time += h

    # Record final state.
    TG, DG, MG, M, E, GL, T = state

    results.append({
        "Time (min)": params.final_time,
        "TG (mol/L)": TG,
        "DG (mol/L)": DG,
        "MG (mol/L)": MG,
        "Methanol (mol/L)": M,
        "FAME (mol/L)": E,
        "Glycerol (mol/L)": GL,
        "Temperature (C)": T - 273.15,
        "FAME Conversion (%)": (
            E / (3.0 * initial_TG)
        ) * 100.0
    })

    return pd.DataFrame(results)


# ==========================================================
# 7. REACTOR SAFETY CHECK
# ==========================================================

def check_temperature_safety(data):
    """
    Check whether the reactor temperature exceeds
    methanol's normal boiling point (64.7 degrees C).
    """

    maximum_temperature = data[
        "Temperature (C)"
    ].max()

    if maximum_temperature >= 64.7:
        return {
            "safe": False,
            "message": (
                "WARNING: Reactor temperature reached or "
                "exceeded methanol's normal boiling point "
                "of 64.7 C. Vapor pressure and pressure "
                "control must be evaluated."
            )
        }

    return {
        "safe": True,
        "message": (
            "Temperature remained below methanol's "
            "normal boiling point at 1 atm."
        )
    }
