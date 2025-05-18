# ref internal research file 'TOP FILE - Bell - Malus LHV/bell3_advanced_success/2b_Paper2_Figure1.py'

import numpy as np
from scipy.integrate import quad, IntegrationWarning
import matplotlib.pyplot as plt
import warnings

# --- Define q_final(L) ---
def q_final(L):
    """Calculates the appropriate stable real root of q^3 - q - L = 0."""
    if np.isclose(L, 0.0):
        return 0.0 # Continuous definition based on root finding for L!=0

    L_crit = 2.0 / (3.0 * np.sqrt(3.0)) # Approx 0.3849
    tol = 1e-9 # Tolerance for floating point comparisons
    abs_L = np.abs(L)

    if abs_L > L_crit + tol:
        discriminant_term = np.maximum(0.0, (L / 2.0)**2 - 1.0 / 27.0)
        sqrt_delta = np.sqrt(discriminant_term)
        A = np.sign(L / 2.0 + sqrt_delta) * np.cbrt(np.abs(L / 2.0 + sqrt_delta))
        B = np.sign(L / 2.0 - sqrt_delta) * np.cbrt(np.abs(L / 2.0 - sqrt_delta))
        q = A + B
    elif abs_L >= L_crit - tol:
         q = np.sign(L) * 2.0 / np.sqrt(3.0)
    else: # abs(L) < L_crit
        arg_arccos = np.clip(L * np.sqrt(27.0) / 2.0, -1.0, 1.0)
        phi = np.arccos(arg_arccos)
        if L > 0:
            q = (2.0 / np.sqrt(3.0)) * np.cos(phi / 3.0)
        else: # L < 0
            # Root corresponding to q -> -1 as L -> 0-
            q = (2.0 / np.sqrt(3.0)) * np.cos(phi / 3.0 + 2 * np.pi / 3.0)
    return q

# --- Define the four Continuous Outcome Functions C(L) ---

def C_qfinal(L):
    """Continuous outcome: The final state q_final itself."""
    return q_final(L)

def C_W_bias(L):
    """Continuous outcome: Work done by bias, C = L * q_final(L)."""
    # Ensure non-negative robustly
    return max(0.0, L * q_final(L))

def C_delta_q(L):
    """Continuous outcome: Shift in min location, C = max(0, |q_final(L)| - q0)."""
    q0 = 1.0 # Magnitude of minimum location at L=0
    return max(0.0, np.abs(q_final(L)) - q0)

def C_delta_E(L):
    """Continuous outcome: Change in potential energy, C = -U(q_final(L))."""
    qf = q_final(L)
    # Potential Energy U(q) = q^4/4 - q^2/2 - L*q
    U_q_final = 0.25 * qf**4 - 0.5 * qf**2 - L * qf
    # Note: This can be positive or negative
    return -U_q_final

# --- Source Distribution ---
def rho(lam):
    return 0.25 * np.abs(np.cos(2.0 * lam))

# --- Integrand Definition using State Shift ---
def integrand_state_shift(lam, s1, s2, C_func):
    """
    Calculates the value inside the integral for E_cont.
    Uses state shift lambda_2 = lambda + pi/2.
    """
    # L1 for particle 1 measured with setting s1
    L1 = np.cos(2.0 * (lam - s1))

    # L2 for particle 2 (lambda_eff = lam + pi/2) measured with setting s2
    # L2 = cos(2 * (lambda_eff - s2)) = cos(2 * (lam + np.pi/2.0 - s2))
    #    = cos(2*lam + np.pi - 2*s2) = -cos(2*lam - 2*s2) = -cos(2*(lam - s2))
    L2 = -np.cos(2.0 * (lam - s2))

    # Get the outcomes using the provided C_func
    C1_val = C_func(L1)
    C2_val = C_func(L2)

    rho_val = rho(lam)

    # Product of outcomes multiplied by probability density
    return rho_val * C1_val * C2_val

# --- Perform Numerical Integration ---
delta_s_values = np.linspace(0, np.pi / 2, 100)
s2 = 0.0 # WLOG

# Store C functions and results
C_functions = {
    "q_final": C_qfinal,
    "W_bias": C_W_bias,
    "Delta_q": C_delta_q,
    "Delta_E": C_delta_E,
}
# Store RAW integration results
E_cont_raw_results = {name: [] for name in C_functions}

# Integration settings
cusp_points = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
integration_limit = 150
integration_epsabs=1e-9
integration_epsrel=1e-9

print("Calculating RAW E_continuous for different C(L) using State Shift...")
# Ignore warnings specifically for this section if they reappear
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=IntegrationWarning)
    for i, delta_s in enumerate(delta_s_values):
        s1 = delta_s + s2
        for name, C_func in C_functions.items():
            result, error = quad(integrand_state_shift, 0, 2 * np.pi,
                                 args=(s1, s2, C_func),
                                 limit=integration_limit,
                                 points=cusp_points,
                                 epsabs=integration_epsabs,
                                 epsrel=integration_epsrel)
            # Store the RAW result - NO artificial minus sign
            E_cont_raw_results[name].append(result)
        if (i+1) % 20 == 0: print(f"  .. completed Δs step {i+1}/{len(delta_s_values)}")

print("Raw integration complete.")

# Convert lists to numpy arrays
for name in E_cont_raw_results:
    E_cont_raw_results[name] = np.array(E_cont_raw_results[name])

# --- Calculate QM Prediction ---
E_qm_values = -np.cos(2.0 * delta_s_values)

# --- Plot Results ---
plt.figure(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(C_functions))) # Get distinct colors

label_map = {
    "q_final": r'$E_{cont}$ (raw, $C=q_{final}$)',
    "W_bias": r'$E_{cont}$ (raw, $C=W_{bias}$)',
    "Delta_q": r"$E_{cont}$ (raw, $C=\Delta q_{min}$)",
    "Delta_E": r"$E_{cont}$ (raw, $C=-\Delta E$)"
}

for i, (name, E_values) in enumerate(E_cont_raw_results.items()):
    plt.plot(delta_s_values / np.pi, E_values, label=label_map.get(name, name), linewidth=2, color=colors[i])

plt.plot(delta_s_values / np.pi, E_qm_values, label=r'$E_{QM}(\Delta s) = -\cos(2\Delta s)$', linestyle='--', color='red', linewidth=2)

plt.xlabel(r'$\Delta s / \pi$')
plt.ylabel('Raw Correlation E')
plt.title('WUT LHV Raw Continuous Correlations (State Shift Model) vs QM')
plt.legend(loc='best')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)

# Determine plot limits dynamically based on all data
all_data_min = min(E_qm_values.min(), min(v.min() for v in E_cont_raw_results.values()))
all_data_max = max(E_qm_values.max(), max(v.max() for v in E_cont_raw_results.values()))
plt.ylim(all_data_min - 0.1, all_data_max + 0.1) # Add padding

plt.show()

# --- Summary Table ---
print("\n--- Summary of RAW Correlation Values (State Shift Model) ---")
print(f"{'Curve':<12} | {'Min Value':<10} | {'Max Value':<10} | {'E(Δs=0)':<10} | {'E(Δs=π/4)':<10} | {'E(Δs=π/2)':<10}")
print(f"-------------|------------|------------|------------|-------------|------------")

# Find indices for specific delta_s values
idx_0 = np.abs(delta_s_values - 0.0).argmin()
idx_pi_4 = np.abs(delta_s_values - np.pi/4.0).argmin()
idx_pi_2 = np.abs(delta_s_values - np.pi/2.0).argmin()

for name, E_values in E_cont_raw_results.items():
    min_val = np.min(E_values)
    max_val = np.max(E_values)
    val_0 = E_values[idx_0]
    val_pi4 = E_values[idx_pi_4]
    val_pi2 = E_values[idx_pi_2]
    print(f"{name:<12} | {min_val:<10.4f} | {max_val:<10.4f} | {val_0:<10.4f} | {val_pi4:<10.4f} | {val_pi2:<10.4f}")

# Add QM for comparison
min_qm = np.min(E_qm_values)
max_qm = np.max(E_qm_values)
val_0_qm = E_qm_values[idx_0]
val_pi4_qm = E_qm_values[idx_pi_4]
val_pi2_qm = E_qm_values[idx_pi_2]
print(f"{'QM Target':<12} | {min_qm:<10.4f} | {max_qm:<10.4f} | {val_0_qm:<10.4f} | {val_pi4_qm:<10.4f} | {val_pi2_qm:<10.4f}")