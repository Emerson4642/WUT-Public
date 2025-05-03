import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Reuse q_final(L) from previous task ---
def q_final(L):
    """Calculates the appropriate stable real root of q^3 - q - L = 0."""
    if np.isclose(L, 0.0):
        return 0.0

    L_crit = 2.0 / (3.0 * np.sqrt(3.0)) # Approx 0.3849
    tol = 1e-9 # Tolerance for float comparison

    if np.abs(L) > L_crit + tol:
        # Case 1: |L| > L_crit (One real root)
        discriminant_term = np.maximum(0.0, (L / 2.0)**2 - 1.0 / 27.0)
        sqrt_delta = np.sqrt(discriminant_term)
        term1_base = L / 2.0 + sqrt_delta
        term2_base = L / 2.0 - sqrt_delta
        A = np.sign(term1_base) * np.cbrt(np.abs(term1_base))
        B = np.sign(term2_base) * np.cbrt(np.abs(term2_base))
        q = A + B
    elif np.abs(np.abs(L) - L_crit) <= tol:
         # Case 2: |L| = L_crit
         q = np.sign(L) * 2.0 / np.sqrt(3.0)
    else: # abs(L) < L_crit
        # Case 3: |L| < L_crit (Three real roots)
        arg_arccos = np.clip(L * np.sqrt(27.0) / 2.0, -1.0, 1.0)
        phi = np.arccos(arg_arccos)
        if L > 0:
            q = (2.0 / np.sqrt(3.0)) * np.cos(phi / 3.0)
        else: # L < 0
            q = (2.0 / np.sqrt(3.0)) * np.cos(phi / 3.0 - 4.0 * np.pi / 3.0)
    return q

# --- Define sgn function ---
def sgn(x):
    """Sign function: 1 if x >= 0, -1 if x < 0."""
    return 1.0 if x >= 0 else -1.0

# --- Define Binary Outcome function O(s, lambda) ---
def O_outcome(s, lam):
    """Calculates the binary outcome O(s, lambda) = sgn(q_final(L))."""
    # Ensure lambda stays within [0, 2pi) effectively if shifted
    # lam = lam % (2 * np.pi) # Modulo not strictly needed if lam in integrand is always base lambda
    L = np.cos(2.0 * (lam - s))
    q = q_final(L)
    return sgn(q)

# --- Define the integrand for E''_reported (State Anticorrelation) ---
def integrand_reported_state(lam, s1, s2):
    """The integrand for E''_reported using state anticorrelation lambda2 = lambda + pi/2."""
    rho = 0.25 * np.abs(np.cos(2.0 * lam))
    
    # Outcome for particle 1 (lambda1 = lambda)
    O1 = O_outcome(s1, lam) 
    
    # Outcome for particle 2 (lambda2 = lambda + pi/2)
    # We can calculate L2 directly: L2 = cos(2*( (lam + pi/2) - s2)) = -cos(2*(lam - s2))
    # Or calculate O2 using the shifted lambda:
    O2 = O_outcome(s2, lam + np.pi / 2.0) # Pass the shifted lambda to the O_outcome function

    return rho * O1 * O2

# --- Perform Numerical Integration ---
delta_s_values = np.linspace(0, np.pi / 2, 100) # Range of relative angles
E_reported_state_values = []
s2 = 0.0 # WLOG

print("Calculating E''_reported (state anticorrelation)...")
for i, delta_s in enumerate(delta_s_values):
    s1 = delta_s + s2
    # Integrate the function from 0 to 2*pi
    result, error = quad(integrand_reported_state, 0, 2 * np.pi, args=(s1, s2), epsabs=1e-8, epsrel=1e-8, limit=200)
    E_reported_state_values.append(result)
    # Simple progress indicator
    if (i+1) % 10 == 0:
        print(f"  .. completed {i+1}/{len(delta_s_values)}")

E_reported_state_values = np.array(E_reported_state_values)
print("Calculation complete.")

# --- Calculate QM Prediction ---
E_qm_values = -np.cos(2.0 * delta_s_values)

# --- Plot Results ---
plt.figure(figsize=(10, 6))
plt.plot(delta_s_values / np.pi, E_reported_state_values, label=r'$E^{\prime\prime}_{reported}(\Delta s)$ (State Anticorr.)', linewidth=2)
plt.plot(delta_s_values / np.pi, E_qm_values, label=r'$E_{QM}(\Delta s) = -\cos(2\Delta s)$', linestyle='--', color='red')
plt.xlabel(r'$\Delta s / \pi$')
plt.ylabel('Correlation E')
plt.title('WUT LHV State Anticorrelation vs Quantum Mechanics')
plt.legend()
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.show()

# --- Analyze Difference/Error ---
print("\nComparison (State Anticorrelation):")
print(f"Minimum E''_reported: {np.min(E_reported_state_values):.4f} at Delta s = {delta_s_values[np.argmin(E_reported_state_values)]/np.pi:.2f} pi")
print(f"Maximum E''_reported: {np.max(E_reported_state_values):.4f} at Delta s = {delta_s_values[np.argmax(E_reported_state_values)]/np.pi:.2f} pi")
print(f"Minimum E_QM: {np.min(E_qm_values):.4f} at Delta s = {delta_s_values[np.argmin(E_qm_values)]/np.pi:.2f} pi")
print(f"Maximum E_QM: {np.max(E_qm_values):.4f} at Delta s = {delta_s_values[np.argmax(E_qm_values)]/np.pi:.2f} pi")

# Check value at Delta s = pi/4 (where QM is 0)
idx_pi_over_4 = np.abs(delta_s_values - np.pi/4).argmin()
print(f"E''_reported at Delta s = pi/4: {E_reported_state_values[idx_pi_over_4]:.4f}")
print(f"E_QM at Delta s = pi/4: {E_qm_values[idx_pi_over_4]:.4f}")

# Calculate Mean Squared Error (MSE)
mse_state = np.mean((E_reported_state_values - E_qm_values)**2)
print(f"\nMean Squared Error (MSE) between E''_reported and E_QM: {mse_state:.6f}")

# --- Check if result matches the previous calculation (Task 3.1b) ---
# (Load or re-run 3.1b result if necessary)
# Let's assume E_reported_values from 3.1b is available
# mse_vs_31b = np.mean((E_reported_state_values - E_reported_values)**2)
# print(f"Mean Squared Error (MSE) between E''_reported (3.1c) and E_reported (3.1b): {mse_vs_31b:.6g}")
# Based on theoretical analysis, this MSE should be very close to zero.