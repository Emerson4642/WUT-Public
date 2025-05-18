# Reference internal file 'TOP FILE - Bell - Malus LHV/bell3_advanced_success/1z_optimizer4_FINAL_WORKING_a_b_k_analysis.py'

import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import minimize # Optimizer added
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colors if needed later
import warnings
import time # To time optimization
import sys # For feedback flushing
import csv # <<< Added for CSV output
import os # <<< Added for creating output directory

# --- High Precision q_final Implementation ---
# (Keep the q_final_cardano function exactly as it was in the first script)
def q_final_cardano(L, a, b):
    """
    Calculates the appropriate stable real root of a*q^3 - b*q - L = 0.
    Uses high precision internal tolerances.
    """
    INTERNAL_TOL = 1e-14
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)): return np.nan
    if a <= INTERNAL_TOL or b < 0: return np.nan
    a = max(a, np.finfo(float).eps)
    if np.isclose(L, 0.0, atol=INTERNAL_TOL): return 0.0
    b_eff = b
    try:
        if np.isclose(b_eff, 0.0, atol=INTERNAL_TOL): L_crit = 0.0
        else:
             term_inside_sqrt = max(3.0 * a, np.finfo(float).eps)
             L_crit = (2.0 * b_eff * np.sqrt(b_eff)) / (3.0 * np.sqrt(term_inside_sqrt))
        p = -b / a; r = -L / a
        tol = INTERNAL_TOL; abs_L = np.abs(L)
        if abs_L > L_crit + tol:
            p_term = (p / 3.0)**3; r_term = (r / 2.0)**2
            discriminant = r_term + p_term
            sqrt_discriminant = np.sqrt(np.maximum(0.0, discriminant))
            term1 = -r / 2.0
            arg1 = term1 + sqrt_discriminant; arg2 = term1 - sqrt_discriminant
            A = np.sign(arg1) * np.cbrt(np.abs(arg1)); B = np.sign(arg2) * np.cbrt(np.abs(arg2))
            q = A + B
        elif abs_L >= L_crit - tol:
             if np.isclose(L_crit, 0.0, atol=INTERNAL_TOL): q = np.cbrt(L/a)
             else: q = np.sign(L) * 2.0 * np.sqrt(b_eff / (3.0 * a))
        else:
            term_p_neg = -p / 3.0
            if term_p_neg <= tol: q = np.cbrt(L/a)
            else:
                prefactor = 2.0 * np.sqrt(term_p_neg)
                denom_term_sqrt = np.sqrt(term_p_neg**3)
                if denom_term_sqrt <= tol: q = np.cbrt(L/a)
                else:
                    cos_arg = np.clip(-r / (2.0 * denom_term_sqrt), -1.0, 1.0)
                    phi = np.arccos(cos_arg)
                    if L > 0: q = prefactor * np.cos(phi / 3.0)
                    else: q = prefactor * np.cos(phi / 3.0 + 2.0 * np.pi / 3.0)
    except (ValueError, TypeError, FloatingPointError) as e: return np.nan
    except Exception as e: return np.nan
    if not np.isfinite(q): return np.nan
    return q

# --- Constants and Scenario Definition ---

# Define the list of values to use for initial 'a' guess and upper bounds
# (Using the numbers provided by the user)
scenario_values = np.array([
    1.0, 1.75, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0,
    200.0, 400.0, 800.0, 1500.0, 2500.0, 5000.0,
    10000.0, 20000.0, 40000.0, 80000.0, 1000000.0 # Corrected last value from 1M to 100k as per example
], dtype=float)

# Define the target difference for the initial 'b' guess
TARGET_A_MINUS_B = np.sqrt(2.0 / 3.0)
print(f"Target a - b difference (for initial guess): sqrt(2/3) = {TARGET_A_MINUS_B:.15f}")

# Define the FIXED lower bound for optimization
LOWER_BOUND_A_B = 0.1

# Output file names
CSV_FILENAME = "optimization_scenario_results.csv"
PLOT_A_MINUS_B_FILENAME = "scenario_a_minus_b.png"
PLOT_SSE_FILENAME = "scenario_sse.png"
OUTPUT_DIR = "optimization_results" # Directory to save files

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

CSV_FILEPATH = os.path.join(OUTPUT_DIR, CSV_FILENAME)
PLOT_A_MINUS_B_FILEPATH = os.path.join(OUTPUT_DIR, PLOT_A_MINUS_B_FILENAME)
PLOT_SSE_FILEPATH = os.path.join(OUTPUT_DIR, PLOT_SSE_FILENAME)


# Define QM target function
def E_QM(delta_s):
    return -np.cos(2 * delta_s)

# Define the range of delta_s for calculations and plotting
delta_s_rad = np.linspace(0, np.pi / 2, 100, dtype=np.float64) # Use float64 for precision

# --- Integrand Definition (using the Cardano solver) ---
# (Keep the integrand_E_cont function exactly as it was in the first script)
def integrand_E_cont(lambda_val, delta_s, a, b):
    rho = 0.25 * np.abs(np.cos(2 * lambda_val))
    L1 = np.cos(2 * (lambda_val - delta_s))
    L2 = -np.cos(2 * lambda_val)
    q1 = q_final_cardano(L1, a, b)
    q2 = q_final_cardano(L2, a, b)
    if np.isnan(q1) or np.isnan(q2): return np.nan
    return rho * q1 * q2

# --- Integration Function (wrapper around quad) ---
# (Keep the calculate_E_cont_point function exactly as it was in the first script)
def calculate_E_cont_point(delta_s, a, b, quad_options=None):
    if quad_options is None:
        quad_options = {
            'limit': 200, 'epsabs': 1e-14, 'epsrel': 1e-14,
            'points': [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
        }
    result, abserr = np.nan, np.nan
    if a <= 0 or b < 0 or not np.isfinite(a) or not np.isfinite(b): return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            result, abserr = quad(
                integrand_E_cont, 0, 2 * np.pi, args=(delta_s, a, b), **quad_options
            )
        if np.isnan(result): pass
    except Exception as e:
        # Suppress excessive error printing during optimization runs
        # print(f"ERROR: Unexpected integration error ds={delta_s:.4f}, a={a:.4f}, b={b:.4f}: {e}")
        result = np.nan
    return result


# --- Calculation Loop (for a full curve) ---
# (Keep the calculate_E_cont_curve function exactly as it was in the first script)
def calculate_E_cont_curve(delta_s_values, a, b, quad_options=None, verbose=False):
    results = []
    # Reduced verbosity inside the loop
    # if verbose: print(f"  Calculating E_cont curve for a={a:.6f}, b={b:.6f}...")
    num_points = len(delta_s_values)
    nan_count = 0
    # start_time = time.time() # Removed timing for individual curve calc
    for i, ds in enumerate(delta_s_values):
        e_cont = calculate_E_cont_point(ds, a, b, quad_options)
        results.append(e_cont)
        if np.isnan(e_cont): nan_count += 1
    # end_time = time.time()
    # if verbose: print(f"  Curve calculation finished in {end_time - start_time:.2f} sec. Found {nan_count} NaN(s).")
    # Only warn if NaNs occurred and optimization is not running (or if verbose)
    if nan_count > 0 and verbose:
        print(f"WARNING: Calc for a={a:.6f}, b={b:.6f} -> {nan_count}/{num_points} NaNs.")
    return np.array(results)

# --- Objective Function for Optimizer ---
# Define QM target values globally once
E_qm_values_global = E_QM(delta_s_rad)

# Define GLOBAL integration options for use INSIDE the OPTIMIZER
integration_options_for_optimizer = {
    'limit': 150, 'epsabs': 1e-13, 'epsrel': 1e-13,
    'points': [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
}

# Globals for feedback during optimization (will be reset each scenario)
obj_func_call_count = 0
best_sse = np.inf
best_params = [np.nan, np.nan]
scenario_start_time = 0 # Renamed from start_time
HIGH_PENALTY = 1e10 # Penalty for invalid points/params

def sse_objective_function(params):
    """ Objective function for the optimizer. Calculates the SSE. """
    global obj_func_call_count, best_sse, best_params, scenario_start_time
    obj_func_call_count += 1
    a, b = params

    # Check if params are within bounds (redundant with L-BFGS-B but good practice)
    # Note: Bounds are handled by the optimizer itself primarily.
    if a < LOWER_BOUND_A_B or b < LOWER_BOUND_A_B or not np.isfinite(a) or not np.isfinite(b):
       # No need for status line here, optimizer handles bounds
       return HIGH_PENALTY

    # Calculate E_cont curve using the specific options for optimization runs
    E_cont_values = calculate_E_cont_curve(delta_s_rad, a, b, integration_options_for_optimizer, verbose=False)

    # Calculate SSE, handling potential NaNs
    valid_mask = ~np.isnan(E_cont_values)
    num_valid = np.sum(valid_mask)
    num_total = len(delta_s_rad)
    status_suffix = ""

    if num_valid == 0:
        sse = HIGH_PENALTY * 100
        status_suffix = f"-> All NaN! SSE = {sse:.2e}"
    else:
        sse = np.sum((E_cont_values[valid_mask] - E_qm_values_global[valid_mask])**2)
        if num_valid < num_total:
            # Adjusted penalty to be less aggressive but still discourage NaNs
            mean_sq_err = sse / num_valid if num_valid > 0 else 1.0
            nan_penalty = (num_total - num_valid) * mean_sq_err * 0.1 # Smaller penalty factor
            sse += nan_penalty
            status_suffix = f"-> SSE = {sse:.10f} ({num_valid}/{num_total} valid, NaN pen)" # Shorter display
        else:
            status_suffix = f"-> SSE = {sse:.10f} ({num_valid}/{num_total} valid)" # Shorter display SSE

    # Update Best Found So Far (only if current SSE is valid and better)
    if np.isfinite(sse) and sse < best_sse:
        best_sse = sse
        best_params = [a, b]

    # Feedback Print (reduced precision for status line)
    elapsed_time = time.time() - scenario_start_time
    status_line = (
        f"Iter: {obj_func_call_count:<5} | T: {elapsed_time:<5.1f}s | "
        f"Curr [a,b]: [{a:<10.4f}, {b:<10.4f}] | "
        f"Best SSE: {best_sse:<15.9f} | " # Shorter display best SSE
        f"{status_suffix}"
    )
    # Limit line length
    max_len = 120
    print(status_line[:max_len] + " ", end='\r')
    sys.stdout.flush()

    return sse if np.isfinite(sse) else HIGH_PENALTY * 10


# --- Main Scenario Loop ---
print("\n--- Starting Optimization Scenario Loop (this script may run for several hours)---")
all_results = [] # List to store results from each scenario

# Loop through each value which defines the scenario's initial 'a' guess and upper bound
for scenario_value in scenario_values:
    initial_a_guess = scenario_value
    upper_bound_for_scenario = scenario_value # Using the same value for upper bound

    # Define bounds for this specific scenario
    # [(min_a, max_a), (min_b, max_b)]
    current_optimizer_bounds = [
        (LOWER_BOUND_A_B, upper_bound_for_scenario),
        (LOWER_BOUND_A_B, upper_bound_for_scenario) # Apply same upper bound to b
    ]

    # Calculate initial guess for b based on a
    initial_b_guess = initial_a_guess - TARGET_A_MINUS_B
    # Ensure initial guess for b is within the bounds (specifically >= lower bound)
    initial_b_guess = max(initial_b_guess, current_optimizer_bounds[1][0])
    # Ensure initial guess for b is not above its upper bound (less likely but possible if a_guess is very small)
    initial_b_guess = min(initial_b_guess, current_optimizer_bounds[1][1])

    current_initial_guess = [initial_a_guess, initial_b_guess]

    print(f"\n--- Running Scenario: Initial a ≈ {initial_a_guess:.2f}, Upper Bound = {upper_bound_for_scenario:.2f} ---")
    print(f"Initial Guess: a = {current_initial_guess[0]:.8f}, b = {current_initial_guess[1]:.8f}")
    print(f"Bounds: a in [{current_optimizer_bounds[0][0]:.2f}, {current_optimizer_bounds[0][1]:.2f}], b in [{current_optimizer_bounds[1][0]:.2f}, {current_optimizer_bounds[1][1]:.2f}]")

    # Reset counters and best values for this scenario run
    obj_func_call_count = 0
    best_sse = np.inf
    best_params = [np.nan, np.nan]
    scenario_start_time = time.time() # Start timer for this scenario

    # Run the optimizer for the current scenario
    optimizer_result = minimize(
        sse_objective_function,
        current_initial_guess,
        method='L-BFGS-B', # Good choice for bound constraints
        bounds=current_optimizer_bounds,
        options={
            'disp': False,      # Use custom feedback
            'ftol': 1e-15,      # Function tolerance (very tight)
            'gtol': 1e-10,      # Gradient tolerance
            'maxiter': 5000     # Max iterations (adjust if needed)
            #'maxfun': 15000    # Max function evaluations (optional)
        }
    )

    optimization_end_time = time.time()
    print() # Newline after status updates
    print(f"--- Scenario Finished (Initial a ≈ {initial_a_guess:.2f}) ---")
    print(f"Optimization time: {optimization_end_time - scenario_start_time:.2f} seconds")
    print(f"Total objective function calls: {obj_func_call_count}")

    # Process and store results for this scenario
    scenario_output = {
        "Initial_A_Guess": initial_a_guess,
        "Upper_Bound": upper_bound_for_scenario,
        "Optimal_A": np.nan,
        "Optimal_B": np.nan,
        "Final_SSE": np.nan,
        "A_minus_B": np.nan,
        "Success": optimizer_result.success,
        "Message": optimizer_result.message.replace('\n', ' ').strip() # Clean message
    }

    if optimizer_result.success:
        found_a, found_b = optimizer_result.x
        final_sse = optimizer_result.fun # Note: This is the penalized SSE from the objective function

        # Recalculate final UNPENALIZED SSE with optimal params for reporting
        final_E_cont_values = calculate_E_cont_curve(delta_s_rad, found_a, found_b, verbose=False)
        valid_mask_final = ~np.isnan(final_E_cont_values)
        num_valid_final = np.sum(valid_mask_final)
        if num_valid_final > 0:
             final_sse_unpenalized = np.sum((final_E_cont_values[valid_mask_final] - E_qm_values_global[valid_mask_final])**2)
        else:
             final_sse_unpenalized = np.nan # Could not calculate unpenalized SSE

        print(f"Optimization successful!")
        print(f"  Found optimal parameters: a = {found_a:.10f}, b = {found_b:.10f}")
        print(f"  Final Objective Value (penalized SSE): {final_sse:.10e}")
        print(f"  Final UNPENALIZED SSE: {final_sse_unpenalized:.10e} ({num_valid_final}/{len(delta_s_rad)} valid pts)")
        cleaned_message = optimizer_result.message.replace('\n', ' ').strip()
        print(f"  Optimizer message: {cleaned_message}")
        scenario_output["Optimal_A"] = found_a
        scenario_output["Optimal_B"] = found_b
        scenario_output["Final_SSE"] = final_sse_unpenalized # Store the unpenalized SSE
        scenario_output["A_minus_B"] = found_a - found_b

    else:
        print(f"Optimization FAILED.")
        cleaned_message = optimizer_result.message.replace('\n', ' ').strip()
        print(f"  Reason: {cleaned_message}")
        # Store last attempted parameters if available
        last_a, last_b = optimizer_result.x if hasattr(optimizer_result, 'x') else [np.nan, np.nan]
        scenario_output["Optimal_A"] = last_a # Store last attempted
        scenario_output["Optimal_B"] = last_b # Store last attempted
        if np.isfinite(last_a) and np.isfinite(last_b):
             scenario_output["A_minus_B"] = last_a - last_b
        # Final_SSE remains NaN if failed

    all_results.append(scenario_output)


# --- Save Results to CSV ---
print(f"\n--- Saving Results to CSV ---")
if not all_results:
    print("No results to save.")
else:
    # Use the keys from the first result dictionary as headers
    csv_headers = list(all_results[0].keys())
    try:
        with open(CSV_FILEPATH, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Successfully saved results to {CSV_FILEPATH}")
    except IOError as e:
        print(f"ERROR: Could not write CSV file to {CSV_FILEPATH}: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during CSV writing: {e}")


# --- Plotting Results ---
print("\n--- Generating Plots ---")

# Filter results for successful runs with valid data for plotting
plot_data = [res for res in all_results if res["Success"] and np.isfinite(res["Optimal_A"]) and np.isfinite(res["Optimal_B"]) and np.isfinite(res["Final_SSE"])]

if not plot_data:
    print("No successful optimization results with valid data to plot.")
else:
    # Extract data for plots
    plot_initial_a = np.array([res["Initial_A_Guess"] for res in plot_data])
    plot_optimal_a = np.array([res["Optimal_A"] for res in plot_data])
    plot_optimal_b = np.array([res["Optimal_B"] for res in plot_data])
    plot_a_minus_b = np.array([res["A_minus_B"] for res in plot_data])
    plot_sse = np.array([res["Final_SSE"] for res in plot_data])

    # Ensure SSE is non-negative for log plot
    plot_sse = np.maximum(plot_sse, 1e-20) # Set a floor for log plot

    try: plt.style.use('seaborn-v0_8-darkgrid')
    except OSError: print("Warning: 'seaborn-v0_8-darkgrid' style not found."); plt.style.use('default')

    # Plot 1: a - b vs Initial a Guess
    plt.figure(figsize=(10, 6))
    plt.semilogx(plot_initial_a, plot_a_minus_b, marker='o', linestyle='-', color='blue', label='Optimized $a - b$')
    plt.axhline(TARGET_A_MINUS_B, color='red', linestyle='--', label=f'Target $\sqrt{{2/3}} \\approx {TARGET_A_MINUS_B:.6f}$')
    plt.xlabel('Initial Guess for $a$ (log scale)')
    plt.ylabel('Optimized $a - b$')
    plt.title('Convergence of $a - b$ Across Scenarios')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(PLOT_A_MINUS_B_FILEPATH)
        print(f"Saved plot: {PLOT_A_MINUS_B_FILEPATH}")
    except Exception as e:
        print(f"Error saving plot {PLOT_A_MINUS_B_FILENAME}: {e}")


    # Plot 2: Final SSE vs Initial a Guess
    plt.figure(figsize=(10, 6))
    plt.loglog(plot_initial_a, plot_sse, marker='s', linestyle=':', color='green', label='Final Unpenalized SSE')
    plt.xlabel('Initial Guess for $a$ (log scale)')
    plt.ylabel('Final Unpenalized SSE (log scale)')
    plt.title('Final Sum of Squared Error (SSE) Across Scenarios')
    # Add reference lines for SSE levels if desired
    plt.axhline(1e-12, color='gray', linestyle=':', alpha=0.7, label='SSE $10^{-12}$')
    plt.axhline(1e-14, color='darkorange', linestyle=':', alpha=0.7, label='SSE $10^{-14}$')
    plt.axhline(1e-16, color='magenta', linestyle=':', alpha=0.7, label='SSE $10^{-16}$')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(PLOT_SSE_FILEPATH)
        print(f"Saved plot: {PLOT_SSE_FILEPATH}")
    except Exception as e:
        print(f"Error saving plot {PLOT_SSE_FILENAME}: {e}")

    # plt.show() # Uncomment if you want plots to display interactively

print("\nScript finished.")