
"""
BIODIESEL REACTOR - RK4 NUMERICAL VERIFICATION

Compares the custom Runge-Kutta 4th Order method
against SciPy's solve_ivp numerical solver.

Verification criteria:
Maximum normalized relative error < 1e-4
"""

import numpy as np

from scipy.integrate import solve_ivp

from reactor import (
    ReactorParameters,
    initial_conditions,
    reactor_model,
    rk4_step
)


# ==========================================================
# 1. VERIFICATION SETTINGS
# ==========================================================

RELATIVE_ERROR_LIMIT = 1e-4

# Test the original step size and two smaller step sizes.
STEP_SIZE_FACTORS = [1.0, 0.5, 0.25]

# High-accuracy settings for the reference solver.
SOLVER_RTOL = 1e-10
SOLVER_ATOL = 1e-12

STATE_NAMES = [
    "Triglyceride (TG)",
    "Diglyceride (DG)",
    "Monoglyceride (MG)",
    "Methanol (M)",
    "FAME (E)",
    "Glycerol (GL)",
    "Temperature (K)"
]


# ==========================================================
# 2. CUSTOM RK4 INTEGRATION
# ==========================================================

def run_custom_rk4(params, step_size):
    """
    Integrate the reactor model using the custom RK4 method.

    Uses the existing rk4_step() function from reactor.py.
    """

    time = 0.0

    state = np.asarray(
        initial_conditions(params),
        dtype=float
    )

    times = [time]
    states = [state.copy()]

    final_time = float(params.final_time)

    while time < final_time - 1e-12:

        # Handle the final partial time step.
        h = min(
            step_size,
            final_time - time
        )

        state = rk4_step(
            reactor_model,
            time,
            state,
            h,
            params
        )

        time += h

        # Avoid floating-point endpoint differences.
        if abs(time - final_time) < 1e-10:
            time = final_time

        times.append(time)
        states.append(state.copy())

    return (
        np.array(times, dtype=float),
        np.array(states, dtype=float)
    )


# ==========================================================
# 3. SCIPY SOLVE_IVP REFERENCE SOLUTION
# ==========================================================

def run_solve_ivp(params, evaluation_times):
    """
    Integrate the same reactor model using solve_ivp.

    DOP853 is used as a high-accuracy reference method.
    """

    initial_state = np.asarray(
        initial_conditions(params),
        dtype=float
    )

    solution = solve_ivp(

        fun=lambda t, y: reactor_model(y, params),

        t_span=(
            0.0,
            float(params.final_time)
        ),

        y0=initial_state,

        method="DOP853",

        t_eval=evaluation_times,

        rtol=SOLVER_RTOL,

        atol=SOLVER_ATOL

    )

    if not solution.success:
        raise RuntimeError(
            "solve_ivp failed: " + solution.message
        )

    if solution.y.shape[1] != len(evaluation_times):
        raise RuntimeError(
            "solve_ivp did not return all requested time points."
        )

    return solution.y.T


# ==========================================================
# 4. RELATIVE ERROR CALCULATION
# ==========================================================

def calculate_relative_errors(rk4_results, reference_results):
    """
    Calculate normalized relative error for each state.

    Each state is normalized by the maximum absolute
    reference value of that state over the simulation.

    This avoids dividing by values close to zero.
    """

    absolute_errors = np.abs(
        rk4_results - reference_results
    )

    reference_scales = np.max(
        np.abs(reference_results),
        axis=0
    )

    reference_scales = np.maximum(
        reference_scales,
        1e-12
    )

    normalized_errors = (
        absolute_errors / reference_scales
    )

    maximum_errors = np.max(
        normalized_errors,
        axis=0
    )

    return maximum_errors


# ==========================================================
# 5. RUN ONE VERIFICATION CASE
# ==========================================================

def verify_step_size(params, step_size):
    """
    Compare RK4 and solve_ivp using the same time points.
    """

    # Custom RK4 solution.
    times, rk4_results = run_custom_rk4(
        params,
        step_size
    )

    # Reference solution from solve_ivp.
    reference_results = run_solve_ivp(
        params,
        times
    )

    # Calculate errors.
    errors = calculate_relative_errors(
        rk4_results,
        reference_results
    )

    maximum_error = float(
        np.max(errors)
    )

    passed = (
        maximum_error < RELATIVE_ERROR_LIMIT
    )

    return {
        "step_size": step_size,
        "times": times,
        "rk4_results": rk4_results,
        "reference_results": reference_results,
        "state_errors": errors,
        "maximum_error": maximum_error,
        "passed": passed
    }


# ==========================================================
# 6. DISPLAY VERIFICATION RESULTS
# ==========================================================

def display_results(result):

    print("\n" + "=" * 65)

    print(
        f"RK4 STEP SIZE: {result['step_size']:.6f} min"
    )

    print("=" * 65)

    print(
        f"{'State Variable':<25}"
        f"{'Maximum Relative Error':>27}"
    )

    print("-" * 65)

    for name, error in zip(
        STATE_NAMES,
        result["state_errors"]
    ):

        print(
            f"{name:<25}"
            f"{error:>27.10e}"
        )

    print("-" * 65)

    print(
        "Overall maximum relative error:",
        f"{result['maximum_error']:.10e}"
    )

    print(
        "Required maximum relative error:",
        f"{RELATIVE_ERROR_LIMIT:.10e}"
    )

    if result["passed"]:

        print("\nRESULT: PASS")

        print(
            "Custom RK4 satisfies the numerical "
            "verification tolerance."
        )

    else:

        print("\nRESULT: NOT YET PASSED")

        print(
            "Reduce the RK4 step size and run "
            "the verification again."
        )


# ==========================================================
# 7. MAIN VERIFICATION PROGRAM
# ==========================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "BIODIESEL REACTOR - RK4 VS SOLVE_IVP VERIFICATION"
    )

    print("=" * 65)

    # Use the same default reactor parameters
    # as the main biodiesel simulation.
    params = ReactorParameters()

    print("\nSIMULATION SETTINGS")
    print("-" * 65)

    print(
        f"Reactor volume: {params.volume} L"
    )

    print(
        f"Initial temperature: "
        f"{params.initial_temperature} C"
    )

    print(
        f"Jacket temperature: "
        f"{params.jacket_temperature} C"
    )

    print(
        f"Methanol-to-oil ratio: "
        f"{params.methanol_ratio}:1"
    )

    print(
        f"Catalyst concentration: "
        f"{params.catalyst_wt_pct} wt%"
    )

    print(
        f"Simulation duration: "
        f"{params.final_time} min"
    )

    print(
        f"Original RK4 step size: "
        f"{params.step_size} min"
    )

    print(
        f"solve_ivp relative tolerance: "
        f"{SOLVER_RTOL}"
    )

    print(
        f"solve_ivp absolute tolerance: "
        f"{SOLVER_ATOL}"
    )

    # ------------------------------------------------------
    # Run verification with multiple RK4 step sizes.
    # ------------------------------------------------------

    all_results = []

    for factor in STEP_SIZE_FACTORS:

        test_step_size = (
            params.step_size * factor
        )

        result = verify_step_size(
            params,
            test_step_size
        )

        all_results.append(result)

        display_results(result)

    # ------------------------------------------------------
    # Final verification summary.
    # ------------------------------------------------------

    print("\n" + "=" * 65)

    print("FINAL VERIFICATION SUMMARY")

    print("=" * 65)

    print(
        f"{'RK4 Step Size':<20}"
        f"{'Maximum Error':<23}"
        f"{'Status':<15}"
    )

    print("-" * 65)

    for result in all_results:

        status = (
            "PASS"
            if result["passed"]
            else "FAILED"
        )

        print(
            f"{result['step_size']:<20.6f}"
            f"{result['maximum_error']:<23.10e}"
            f"{status:<15}"
        )

    # ------------------------------------------------------
    # Overall status.
    # ------------------------------------------------------

    passing_results = [
        result
        for result in all_results
        if result["passed"]
    ]

    print("\n" + "=" * 65)

    if passing_results:

        best_result = min(
            passing_results,
            key=lambda result: result["step_size"]
        )

        print("OVERALL RESULT: PASS")

        print(
            "A tested RK4 step size satisfies "
            "the required numerical tolerance."
        )

        print(
            "Smallest passing step size:",
            f"{best_result['step_size']:.6f} min"
        )

        print(
            "Maximum relative error:",
            f"{best_result['maximum_error']:.10e}"
        )

    else:

        print("OVERALL RESULT: NOT YET PASSED")

        print(
            "None of the tested step sizes reached "
            "the required relative error."
        )

        print(
            "Try reducing the RK4 step size further."
        )

    print("=" * 65)
