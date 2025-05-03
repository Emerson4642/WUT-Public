import numpy as np
from scipy.integrate import quad, IntegrationWarning
# from scipy.optimize import minimize # Keep for context if needed later - Not used here
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colors
import warnings
import sys # For exiting if no tests selected

# --- Use the provided older q_final implementation ---
# [q_final_cardano function remains unchanged - omitted for brevity]
def q_final_cardano(L, a, b):
    """
    Calculates the appropriate stable real root of a*q^3 - b*q - L = 0.
    Based on the provided previous working version (Cardano's method).
    """
    # --- Parameter Validation ---
    if a <= 0 or b < 0:
        return np.nan
    a = max(a, 1e-15)

    # --- Handle L=0 case ---
    if np.isclose(L, 0.0, atol=1e-12):
        return 0.0

    # --- Cardano's Method Calculations ---
    b_eff = max(b, 0.0)
    try:
        if b_eff == 0:
            L_crit = 0.0
        else:
             term_inside_sqrt = 3.0 * a
             if term_inside_sqrt <= 0:
                 L_crit = np.inf
             else:
                 L_crit = (2.0 * b_eff * np.sqrt(b_eff)) / (3.0 * np.sqrt(term_inside_sqrt))

        p = -b / a
        r = -L / a
        tol = 1e-9
        abs_L = np.abs(L)

        if abs_L > L_crit + tol:
            discriminant = (r / 2.0)**2 + (p / 3.0)**3
            sqrt_discriminant = np.sqrt(np.maximum(0.0, discriminant))
            term1 = -r / 2.0
            A = np.sign(term1 + sqrt_discriminant) * np.cbrt(np.abs(term1 + sqrt_discriminant))
            B = np.sign(term1 - sqrt_discriminant) * np.cbrt(np.abs(term1 - sqrt_discriminant))
            q = A + B
        elif abs_L >= L_crit - tol:
            q = np.sign(L) * 2.0 * np.sqrt(b_eff / (3.0 * a))
        else:
            cos_arg_denom_term = -(p/3.0)**3
            if cos_arg_denom_term <= 0:
                 q = np.cbrt(L/a) # Fallback
            else:
                cos_arg = np.clip(-r / (2.0 * np.sqrt(cos_arg_denom_term)), -1.0, 1.0)
                phi = np.arccos(cos_arg)
                prefactor = 2.0 * np.sqrt(b_eff / (3.0 * a))
                if L > 0:
                    q = prefactor * np.cos(phi / 3.0)
                else: # L < 0
                    q = prefactor * np.cos(phi / 3.0 + 2.0 * np.pi / 3.0)
    except (ValueError, TypeError) as e:
         # print(f"Warning: Math error in q_final_cardano for L={L}, a={a}, b={b}: {e}") # Keep silent unless debugging
         return np.nan
    except Exception as e:
        print(f"Warning: Unexpected error in q_final_cardano for L={L}, a={a}, b={b}: {e}")
        return np.nan

    # Sign check (optional, can be verbose)
    # if not np.isclose(L, 0.0) and not np.isclose(np.sign(q), np.sign(L)):
    #     # print(f"Warning: Sign mismatch q_final_cardano! L={L:.3e}, q={q:.3e}, a={a}, b={b}")
    #     pass

    if not np.isfinite(q):
        # print(f"Warning: Non-finite result q={q} in q_final_cardano for L={L}, a={a}, b={b}") # Keep silent unless debugging
        return np.nan
    return q


# --- Constants and Available Test Parameters ---
ALL_PARAMS_TO_TEST = {
    "Baseline": {"a": 1.0, "b": 1.0},
    "User Guess": {"a": 21.981997116, "b": 21.166928086},
    "Found Opt": {"a": 2.183, "b": 1.231},
}

# --- User Selection of Tests ---
# [User selection code remains unchanged - omitted for brevity]
print("Available parameter sets (tests):")
available_test_names = list(ALL_PARAMS_TO_TEST.keys())
for i, name in enumerate(available_test_names):
    params = ALL_PARAMS_TO_TEST[name]
    print(f"  {i+1}: {name} (a={params['a']:.3f}, b={params['b']:.3f})")

selected_test_names = set()
while True:
    try:
        choice = input("Enter numbers of tests to run (e.g., '1', '1,3', '1 2 3'), or leave blank to exit: ")
        if not choice.strip():
             print("No tests selected. Exiting.")
             sys.exit() # Exit if nothing is chosen

        # Allow comma or space separation
        chosen_indices = [int(x.strip()) - 1 for x in choice.replace(',', ' ').split()]

        valid_selection = True
        temp_selected_names = set()
        for index in chosen_indices:
            if 0 <= index < len(available_test_names):
                temp_selected_names.add(available_test_names[index])
            else:
                print(f"Error: Invalid choice '{index + 1}'. Please use numbers between 1 and {len(available_test_names)}.")
                valid_selection = False
                break # Exit inner loop on first error

        if valid_selection:
             selected_test_names = temp_selected_names
             if not selected_test_names: # Handle case where only invalid numbers were entered
                  print("No valid tests selected. Exiting.")
                  sys.exit()
             break # Exit while loop if selection is valid
        # Otherwise, loop again to ask for input

    except ValueError:
        print("Invalid input. Please enter numbers separated by spaces or commas.")
    except Exception as e:
        print(f"An unexpected error occurred during selection: {e}")
        sys.exit()

print("\nSelected tests to run:", ", ".join(sorted(list(selected_test_names))))
PARAMS_TO_RUN = {name: ALL_PARAMS_TO_TEST[name] for name in selected_test_names}


# --- Integrand Definition (using the Cardano solver) ---
# [integrand_E_cont function remains unchanged - omitted for brevity]
def integrand_E_cont(lambda_val, delta_s, a, b):
    """
    The integrand for the E_continuous calculation.
    Uses q_final_cardano.
    """
    rho = 0.25 * np.abs(np.cos(2 * lambda_val))
    L1 = np.cos(2 * (lambda_val - delta_s))
    L2 = -np.cos(2 * lambda_val)

    q1 = q_final_cardano(L1, a, b)
    q2 = q_final_cardano(L2, a, b)

    if np.isnan(q1) or np.isnan(q2):
        return np.nan

    return rho * q1 * q2


# --- Integration Function ---
# [calculate_E_cont_point function remains unchanged - omitted for brevity]
def calculate_E_cont_point(delta_s, a, b, quad_options=None):
    """
    Calculates E_cont for a single delta_s value using scipy.integrate.quad.
    """
    if quad_options is None:
        quad_options = {
            'limit': 150,
            'epsabs': 1e-9,
            'epsrel': 1e-9,
            'points': [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
        }

    result = np.nan
    abserr = np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=IntegrationWarning)
            result, abserr = quad(
                integrand_E_cont,
                0, 2 * np.pi,
                args=(delta_s, a, b),
                **quad_options
            )
        if np.isnan(result):
             # print(f"Warning: quad returned NaN for delta_s={delta_s:.4f}, a={a}, b={b}.") # Keep silent
             pass
    except Exception as e:
        print(f"Error during integration for delta_s={delta_s:.4f}, a={a}, b={b}: {e}")
        result = np.nan
    return result


# --- Calculation Loop ---
# [calculate_E_cont_curve function remains unchanged - omitted for brevity]
def calculate_E_cont_curve(delta_s_values, a, b, quad_options=None):
    """Calculates E_cont for a range of delta_s values."""
    results = []
    print(f"\nCalculating E_cont for a={a:.3f}, b={b:.3f} using Cardano q_final...")
    num_points = len(delta_s_values)
    nan_count = 0
    for i, ds in enumerate(delta_s_values):
        e_cont = calculate_E_cont_point(ds, a, b, quad_options)
        results.append(e_cont)
        if np.isnan(e_cont):
            nan_count += 1
        # Progress print
        # if (i + 1) % 10 == 0 or i == num_points - 1:
        #      print(f"  ... completed {i+1}/{num_points} points (NaNs so far: {nan_count}).") # Optional: Reduce verbosity

    print(f"Calculation finished for a={a:.3f}, b={b:.3f}. ({nan_count} NaN values out of {num_points})")
    # if nan_count > 0:
    #     print(f"WARNING: Calculation resulted in {nan_count} NaN values out of {num_points} points.") # Already printed above
    return np.array(results)


# Define QM target function
def E_QM(delta_s):
    return -np.cos(2 * delta_s)

# Define the range of delta_s
delta_s_rad = np.linspace(0, np.pi / 2, 100) # 100 points from 0 to pi/2

# Integration options
integration_options = {
    'limit': 150,
    'epsabs': 1e-9,
    'epsrel': 1e-9,
    'points': [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
}

# --- Calculate ONLY for selected parameter sets ---
E_cont_results = {}
if not PARAMS_TO_RUN:
     print("\nNo tests selected to run calculations for.")
else:
    for name, params in PARAMS_TO_RUN.items():
        a = params["a"]
        b = params["b"]
        E_cont_results[name] = calculate_E_cont_curve(delta_s_rad, a, b, integration_options)

# --- ALWAYS Calculate QM Target ---
E_qm_values = E_QM(delta_s_rad)

# --- Calculate SSE ONLY for selected cases ---
SSE_results = {}
if E_cont_results: # Only proceed if some tests were run
    print("\n--- Sum of Squared Errors (SSE) vs QM (for selected tests) ---")
    for name, E_cont_values in E_cont_results.items():
        params = PARAMS_TO_RUN[name]
        a = params['a']
        b = params['b']
        valid_mask = ~np.isnan(E_cont_values)
        num_valid = np.sum(valid_mask)
        if num_valid == 0:
            sse = np.nan
            print(f"{name:>12} (a={a:.3f}, b={b:.3f}): No valid points.")
        elif num_valid < len(delta_s_rad):
            sse = np.sum((E_cont_values[valid_mask] - E_qm_values[valid_mask])**2)
            print(f"{name:>12} (a={a:.3f}, b={b:.3f}): SSE={sse:.5f} (over {num_valid} valid points)")
        else:
            sse = np.sum((E_cont_values - E_qm_values)**2)
            print(f"{name:>12} (a={a:.3f}, b={b:.3f}): SSE={sse:.5f}")
        SSE_results[name] = sse


# --- Plotting ---
plt.figure(figsize=(10, 7))

# *** Define Custom Styles and Default Colors ***
custom_styles = {
    "Baseline": {"color": "blue", "linestyle": "-", "marker": '.'}
    # Add more entries here if you want to customize other specific tests
    # e.g., "User Guess": {"color": "red", "linestyle": "--", "marker": 'o'}
}

# Define a list of colors for tests *not* in custom_styles
# Using tab10 colors, skipping blue (index 0) as it's reserved for Baseline
default_colors = plt.cm.tab10.colors[1:] # Starts with orange, green, red, etc.
# You could use a different colormap or a manually defined list:
# default_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] # orange, green, red, purple, brown...

# ALWAYS Plot QM Target first
plt.plot(np.degrees(delta_s_rad), E_qm_values, 'k--', linewidth=2, label='QM Target: $-cos(2Δs)$')

# Plot ONLY selected LHV results
if E_cont_results:
    default_color_index = 0 # Index for assigning default colors

    # Iterate through the selected tests that were actually run
    for name, E_cont_values in E_cont_results.items():
        params = PARAMS_TO_RUN[name]
        a = params['a']
        b = params['b']
        sse = SSE_results.get(name, np.nan)
        label = f'{name} (a={a:.3f}, b={b:.3f}, SSE={sse:.4f})' if not np.isnan(sse) else f'{name} (a={a:.3f}, b={b:.3f}, SSE=NaN)'

        # --- Determine Plot Style ---
        if name in custom_styles:
            style_kwargs = custom_styles[name].copy() # Use the custom style
        else:
            # Get the next default color, cycling if necessary
            plot_color = default_colors[default_color_index % len(default_colors)]
            style_kwargs = {"color": plot_color, "linestyle": "-", "marker": '.'} # Default style for others
            default_color_index += 1 # Only increment when using a default color

        # Add common markersize
        style_kwargs['markersize'] = 4

        # --- Plot the data ---
        valid_mask = ~np.isnan(E_cont_values)
        invalid_count = len(delta_s_rad) - np.sum(valid_mask)

        current_color = style_kwargs['color'] # Get the determined color for NaN markers

        if np.any(valid_mask):
            plt.plot(np.degrees(delta_s_rad[valid_mask]), E_cont_values[valid_mask],
                     label=label, **style_kwargs) # Use **kwargs to apply style

        # Plot NaNs using the same color but with 'x' marker
        if invalid_count > 0:
             plt.plot(np.degrees(delta_s_rad[~valid_mask]), np.zeros(invalid_count),
                      marker='x', markersize=8, linestyle='None', # No line for NaNs
                      color=current_color, # Use the same color as the main line
                      label=f'NaNs ({invalid_count} pts, {name})')

plt.xlabel('Relative Angle Δs (degrees)')
plt.ylabel('Correlation E(Δs)')
plt.title('Continuous Correlation E(Δs) for WUT LHV Model vs QM (Cardano q_final)')
plt.legend(fontsize=9)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(-1.5, 1.5) # Keep fixed Y-axis
plt.tight_layout()
plt.show()


# --- Optional: Summary Table ONLY for selected tests ---
# [Summary table code remains unchanged - omitted for brevity]
if E_cont_results:
    print("\n--- Summary of Correlation Extrema (Raw, C=q_final_cardano, for selected tests) ---")
    print(f"{'Name':<12} | {'Parameter a':<12} | {'Parameter b':<12} | {'Min Value':<12} | {'Max Value':<12} | {'Amplitude':<12}")
    print(f"-------------|--------------|--------------|--------------|--------------|--------------")

    for name, E_cont_values in E_cont_results.items():
        params = PARAMS_TO_RUN[name]
        a = params['a']
        b = params['b']
        valid_mask = ~np.isnan(E_cont_values)
        if np.any(valid_mask):
            min_val = np.min(E_cont_values[valid_mask])
            max_val = np.max(E_cont_values[valid_mask])
            amplitude = 0.5 * (max_val - min_val)
        else:
            min_val, max_val, amplitude = np.nan, np.nan, np.nan

        print(f"{name:<12} | {a:<12.3f} | {b:<12.3f} | {min_val:<12.4f} | {max_val:<12.4f} | {amplitude:<12.4f}")

# ALWAYS Add QM for comparison in the summary table
min_qm = np.min(E_qm_values)
max_qm = np.max(E_qm_values)
amplitude_qm = 0.5 * (max_qm - min_qm)
# Only print the header if we printed the table for selected tests
if E_cont_results:
    print(f"{'QM Target':<12} | {'N/A':<12} | {'N/A':<12} | {min_qm:<12.4f} | {max_qm:<12.4f} | {amplitude_qm:<12.4f}")
else:
    # If no tests ran, maybe print just the QM info standalone or nothing more
    print("\n--- QM Target Extrema ---")
    print(f"{'Name':<12} | {'Min Value':<12} | {'Max Value':<12} | {'Amplitude':<12}")
    print(f"-------------|--------------|--------------|--------------")
    print(f"{'QM Target':<12} | {min_qm:<12.4f} | {max_qm:<12.4f} | {amplitude_qm:<12.4f}")