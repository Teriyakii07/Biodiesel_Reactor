
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from reactor import ReactorParameters, run_simulation

from analysis import (
    newton_raphson,
    sensitivity_analysis,
    find_optimal_conditions
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Biodiesel Reactor Simulator",
    page_icon="⚗️",
    layout="wide"
)





# =========================================================
# CUSTOM DESIGN
# =========================================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        color: #163a5f;
        font-weight: 700;
    }

    h2, h3 {
        color: #244d70;
    }

    /* Simulation Results metric cards only */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e1e8ef;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-size: 14px !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: #163a5f !important;
        font-size: 32px !important;
        font-weight: 700 !important;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)





# =========================================================
# HEADER
# =========================================================

st.title("⚗️ Biodiesel Reactor Simulator")

st.markdown(
    "### 50-Liter Pilot-Scale Batch Reactor"
)

st.caption(
    "Soybean Oil Transesterification | Custom RK4 Numerical Method"
)

st.divider()


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "simulation": None,
    "parameters": None,
    "root_result": None,
    "sensitivity": None,
    "optimization": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# SIDEBAR - REACTOR INPUTS
# =========================================================

st.sidebar.title("Reactor Controls")

st.sidebar.markdown("Adjust the operating conditions below.")

initial_temp = st.sidebar.slider(
    "Initial Temperature (°C)",
    min_value=45.0,
    max_value=65.0,
    value=55.0,
    step=1.0
)

jacket_temp = st.sidebar.slider(
    "Jacket Temperature (°C)",
    min_value=40.0,
    max_value=70.0,
    value=55.0,
    step=1.0
)

methanol_ratio = st.sidebar.slider(
    "Methanol-to-Oil Ratio",
    min_value=3.0,
    max_value=12.0,
    value=6.0,
    step=0.5
)

catalyst = st.sidebar.slider(
    "Catalyst Concentration (wt%)",
    min_value=0.1,
    max_value=1.0,
    value=0.2,
    step=0.05
)

step_size = st.sidebar.select_slider(
    "RK4 Step Size (min)",
    options=[0.1, 0.05, 0.025],
    value=0.1
)

st.sidebar.divider()

run_button = st.sidebar.button(
    "▶ Run Simulation",
    type="primary",
    use_container_width=True
)

st.sidebar.caption(
    "Simulation duration: 120 minutes"
)


# =========================================================
# RUN SIMULATION
# =========================================================

if run_button:

    try:

        params = ReactorParameters(
            initial_temperature=initial_temp,
            jacket_temperature=jacket_temp,
            methanol_ratio=methanol_ratio,
            catalyst_wt_pct=catalyst,
            step_size=step_size
        )

        with st.spinner("Simulating biodiesel production..."):

            results = run_simulation(params)

            st.session_state.simulation = results
            st.session_state.parameters = params

            # Clear previous analysis when inputs change
            st.session_state.root_result = None
            st.session_state.sensitivity = None
            st.session_state.optimization = None

        st.success("Simulation completed successfully!")

    except Exception as error:

        st.error(f"Simulation error: {error}")


# =========================================================
# MAIN DASHBOARD
# =========================================================

if st.session_state.simulation is None:

    st.info(
        "Adjust the reactor parameters in the sidebar "
        "and click Run Simulation to begin."
    )

    st.subheader("About the Simulator")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Reaction Model")
        st.write(
            "Three-step reversible transesterification "
            "of soybean oil using methanol and NaOH catalyst."
        )

    with col2:
        st.markdown("#### Numerical Method")
        st.write(
            "Custom fourth-order Runge-Kutta (RK4) "
            "method for solving mass and energy balances."
        )

    with col3:
        st.markdown("#### Analysis Tools")
        st.write(
            "Newton-Raphson root finding, sensitivity "
            "analysis, and operating condition optimization."
        )

    st.divider()

    st.subheader("Reaction Process")

    st.markdown("""
    The biodiesel production process consists of three
    reversible transesterification reactions:

    1. Triglyceride (TG) + Methanol → Diglyceride (DG) + FAME
    2. Diglyceride (DG) + Methanol → Monoglyceride (MG) + FAME
    3. Monoglyceride (MG) + Methanol → Glycerol (GL) + FAME
    """)

else:

    df = st.session_state.simulation
    params = st.session_state.parameters

    # =====================================================
    # RESULTS OVERVIEW
    # =====================================================

    st.header("Simulation Results")

    final_row = df.iloc[-1]

    final_yield = float(
        final_row["FAME Conversion (%)"]
    )

    final_temp = float(
        final_row["Temperature (C)"]
    )

    max_temp = float(
        df["Temperature (C)"].max()
    )

    duration = float(
        df["Time (min)"].iloc[-1]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Final FAME Conversion",
        f"{final_yield:.2f}%"
    )

    col2.metric(
        "Final Temperature",
        f"{final_temp:.2f} °C"
    )

    col3.metric(
        "Maximum Temperature",
        f"{max_temp:.2f} °C"
    )

    col4.metric(
        "Simulation Duration",
        f"{duration:.1f} min"
    )

    # Yield status
    if final_yield >= 96.5:
        st.success(
            "The simulation reached the required "
            "96.5% FAME conversion target."
        )
    else:
        st.warning(
            f"The simulation did not reach the 96.5% target. "
            f"Final conversion: {final_yield:.2f}%."
        )

    # Temperature safety notice
    if max_temp >= 64.7:

        st.warning(
            "Temperature reached or exceeded methanol's "
            "normal boiling point of 64.7°C at 1 atm. "
            "Review reactor pressure, vapor handling, "
            "and safety requirements."
        )

    else:

        st.success(
            "Reactor temperature remained below "
            "methanol's normal boiling point of 64.7°C."
        )

    st.divider()

    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Reaction Profile",
        "🌡️ Temperature",
        "🎯 Root Finding",
        "📈 Sensitivity & Optimization",
        "📥 Export Data"
    ])

    # =====================================================
    # TAB 1 - CONCENTRATION PROFILE
    # =====================================================

    with tab1:

        st.subheader("Component Concentration Profile")

        st.write(
            "Changes in reactant and product concentrations "
            "throughout the reaction."
        )

        # Find available concentration columns
        aliases = {
            "Triglyceride (TG)": ["TG"],
            "Diglyceride (DG)": ["DG"],
            "Monoglyceride (MG)": ["MG"],
            "Methanol (M)": ["Methanol", "M"],
            "FAME (E)": ["FAME", "E"],
            "Glycerol (GL)": ["Glycerol", "GL"]
        }

        plot_columns = {}

        for label, possible_names in aliases.items():

            for column in df.columns:

                column_lower = str(column).lower().strip()

                for name in possible_names:

                    name_lower = name.lower()

                    if (
                        column_lower == name_lower
                        or column_lower.startswith(name_lower + " ")
                        or column_lower.startswith(name_lower + "(")
                    ):
                        plot_columns[label] = column
                        break

                if label in plot_columns:
                    break

        if plot_columns:

            fig, ax = plt.subplots(figsize=(11, 5))

            for label, column in plot_columns.items():

                ax.plot(
                    df["Time (min)"],
                    df[column],
                    label=label,
                    linewidth=2
                )

            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Concentration (mol/L)")
            ax.set_title("Biodiesel Reaction Progress")
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

        else:

            st.warning(
                "No recognized concentration columns were found. "
                "Available columns are shown below."
            )

            st.write(list(df.columns))

        st.subheader("Reaction Data Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    # =====================================================
    # TAB 2 - TEMPERATURE PROFILE
    # =====================================================

    with tab2:

        st.subheader("Reactor Temperature Profile")

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.plot(
            df["Time (min)"],
            df["Temperature (C)"],
            linewidth=2.5,
            label="Reactor Temperature"
        )

        ax.axhline(
            y=64.7,
            linestyle="--",
            linewidth=1.5,
            label="Methanol Boiling Point (64.7°C)"
        )

        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature (°C)")
        ax.set_title("Temperature vs. Reaction Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)

        col1.metric(
            "Initial Temperature",
            f"{params.initial_temperature:.2f} °C"
        )

        col2.metric(
            "Maximum Temperature",
            f"{max_temp:.2f} °C"
        )

        st.caption(
            "The 64.7°C reference is methanol's normal boiling "
            "point at atmospheric pressure. It is not a reactor "
            "pressure safety limit."
        )

    # =====================================================
    # TAB 3 - ROOT FINDING
    # =====================================================

    with tab3:

        st.subheader("Newton-Raphson Root Finding")

        st.write(
            "Calculate the reaction holding time required "
            "to reach the target FAME conversion of 96.5%."
        )

        if st.button(
            "Calculate Required Holding Time",
            key="root_button"
        ):

            try:

                with st.spinner("Calculating holding time..."):

                    root_result = newton_raphson(params)

                    st.session_state.root_result = root_result

            except Exception as error:

                st.error(f"Root finding error: {error}")

        if st.session_state.root_result is not None:

            result = st.session_state.root_result

            st.divider()

            if result.get("converged", False):

                st.success("Newton-Raphson converged successfully.")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Required Holding Time",
                    f"{result['holding_time']:.4f} min"
                )

                col2.metric(
                    "FAME Conversion",
                    f"{result['conversion']:.4f}%"
                )

                col3.metric(
                    "Iterations",
                    str(result["iterations"])
                )

            else:

                st.warning(
                    result.get(
                        "message",
                        "The target conversion was not reached."
                    )
                )

                if result.get("conversion") is not None:

                    st.metric(
                        "Achieved Conversion",
                        f"{result['conversion']:.4f}%"
                    )

                if result.get("holding_time") is not None:

                    st.metric(
                        "Estimated Holding Time",
                        f"{result['holding_time']:.4f} min"
                    )

            st.caption(
                f"Method: {result.get('method', 'Newton-Raphson')}"
            )

    # =====================================================
    # TAB 4 - SENSITIVITY AND OPTIMIZATION
    # =====================================================

    with tab4:

        st.subheader("Sensitivity Analysis")

        st.write(
            "Evaluate the effects of methanol-to-oil ratio "
            "and initial temperature on FAME yield and energy cost."
        )

        if st.button(
            "Run Sensitivity Analysis",
            key="sensitivity_button"
        ):

            try:

                with st.spinner("Running sensitivity analysis..."):

                    sensitivity = sensitivity_analysis(params)

                    st.session_state.sensitivity = pd.DataFrame(
                        sensitivity
                    )

                    st.session_state.optimization = None

            except Exception as error:

                st.error(
                    f"Sensitivity analysis error: {error}"
                )

        if st.session_state.sensitivity is not None:

            sensitivity_df = st.session_state.sensitivity

            st.subheader("Sensitivity Results")

            st.dataframe(
                sensitivity_df,
                use_container_width=True
            )

            st.divider()

            st.subheader("Operating Condition Optimization")

            st.write(
                "Find the best tested operating condition that "
                "meets the minimum FAME conversion requirement."
            )

            if st.button(
                "Optimize Operating Conditions",
                key="optimization_button"
            ):

                try:

                    with st.spinner("Finding optimal conditions..."):

                        optimization = find_optimal_conditions(
                            sensitivity_df
                        )

                        st.session_state.optimization = optimization

                except Exception as error:

                    st.error(
                        f"Optimization error: {error}"
                    )

        if st.session_state.optimization is not None:

            optimization = st.session_state.optimization

            st.divider()

            st.subheader("Optimization Results")

            if isinstance(optimization, dict):

                col1, col2, col3 = st.columns(3)

                display_fields = [
                    ("methanol_ratio", "Methanol Ratio"),
                    ("initial_temperature", "Initial Temperature"),
                    ("fame_yield", "FAME Yield"),
                    ("energy_cost", "Energy Cost"),
                    ("maximum_temperature", "Maximum Temperature")
                ]

                for index, (key, label) in enumerate(display_fields):

                    if key in optimization:

                        value = optimization[key]

                        if isinstance(value, (int, float)):

                            if key == "energy_cost":
                                display_value = f"₱{value:.2f}"
                            elif key == "initial_temperature" or key == "maximum_temperature":
                                display_value = f"{value:.2f} °C"
                            elif key == "fame_yield":
                                display_value = f"{value:.2f}%"
                            else:
                                display_value = f"{value:.2f}"

                        else:
                            display_value = str(value)

                        with [col1, col2, col3][index % 3]:

                            st.metric(
                                label,
                                display_value
                            )

                if "status" in optimization:

                    st.info(optimization["status"])

            else:

                st.write(optimization)

            st.caption(
                "Optimization identifies the best condition among "
                "the tested sensitivity-analysis cases."
            )

    # =====================================================
    # TAB 5 - EXPORT
    # =====================================================

    with tab5:

        st.subheader("Export Simulation Results")

        st.write(
            "Download the simulation data for further analysis "
            "or inclusion in your project report."
        )

        csv_data = df.to_csv(index=False)

        st.download_button(
            "⬇ Download Simulation Results (CSV)",
            data=csv_data,
            file_name="biodiesel_simulation.csv",
            mime="text/csv",
            use_container_width=True
        )

        excel_buffer = BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            df.to_excel(
                writer,
                sheet_name="Simulation",
                index=False
            )

            if st.session_state.sensitivity is not None:

                st.session_state.sensitivity.to_excel(
                    writer,
                    sheet_name="Sensitivity",
                    index=False
                )

        st.download_button(
            "⬇ Download Simulation Results (Excel)",
            data=excel_buffer.getvalue(),
            file_name="biodiesel_simulation.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )

        st.divider()

        st.subheader("Complete Simulation Data")

        st.dataframe(
            df,
            use_container_width=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Biodiesel Reactor Simulator | Custom RK4 Model | "
    "Educational Simulation"
)
