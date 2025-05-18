#ref initial paper figure 4
#ref internal files 'TOP FILE - Bell - Malus LHV/Bell1_Validation/model_c_paper1_figure5.py'

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- Define the components based on the PROPOSED solution ---

def rho_proposed(lambda_val):
  """Proposed hidden variable distribution rho(lambda)."""
  # rho(lambda) = (1/4) * |cos(2*lambda)|
  return 0.25 * np.abs(np.cos(2.0 * lambda_val))

def K_kernel(lambda_val, delta_s):
  """Calculates the kernel K(lambda, delta_s)."""
  # K = sgn[cos(2*(lambda - delta_s))] * sgn[cos(2*lambda)]
  sign1 = np.sign(np.cos(2.0 * (lambda_val - delta_s)))
  sign2 = np.sign(np.cos(2.0 * lambda_val))
  # Ensure result is 0 if either term is 0 (measure zero points)
  return sign1 * sign2

def integrand_proposed(lambda_val, delta_s):
  """The full function to be integrated: rho_proposed * K_kernel."""
  return rho_proposed(lambda_val) * K_kernel(lambda_val, delta_s)

# --- Perform numerical integration over a range of Delta s ---

# Define the range of angle differences to test
delta_s_values = np.linspace(0, np.pi, 100) # Test from 0 to pi

# Store the numerical integration results
numerical_results = []
integration_errors = []

print("Starting numerical integration with rho = (1/4)|cos(2*lambda)|...")
for ds in delta_s_values:
  # quad integrates from 0 to 2*pi
  result, error = quad(integrand_proposed, 0, 2 * np.pi, args=(ds,), limit=200) # Increased limit for potentially complex integrand
  numerical_results.append(result)
  integration_errors.append(error)

print("Integration complete.")

# Convert results to numpy array
numerical_results = np.array(numerical_results)
integration_errors = np.array(integration_errors)

# --- Calculate the expected analytical result ---
expected_results = np.cos(2.0 * delta_s_values)

# --- Compare and Plot ---

# Check if results are close
tolerance = 1e-6 # Set a tolerance for comparison
are_close = np.allclose(numerical_results, expected_results, atol=tolerance)
print(f"\nNumerical results close to expected cos(2*Delta s) within tolerance {tolerance}? {are_close}")
max_diff = np.max(np.abs(numerical_results - expected_results))
print(f"Maximum absolute difference: {max_diff:.2e}")
max_est_err = np.max(integration_errors)
print(f"Maximum estimated integration error: {max_est_err:.2e}")


# Create the plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                       gridspec_kw={'height_ratios': [3, 1]}) # Give more space to the main plot

# Top plot: Numerical vs. Expected
axs[0].plot(delta_s_values, expected_results, label=r'Target: $\cos(2\Delta s)$', linewidth=3, linestyle='--', color='black', zorder=2)
axs[0].plot(delta_s_values, numerical_results, label=r'Numerical Integral with $\rho = \frac{1}{4}|\cos(2\lambda)|$', linewidth=2, linestyle='-', color='red', marker='o', markersize=3, alpha=0.8, zorder=3)
axs[0].set_ylabel('Integral Value E(Δs)')
axs[0].set_title(r'Numerical Verification: $\int \rho(\lambda) K(\lambda, \Delta s) d\lambda$ vs $\cos(2\Delta s)$')
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(-1.1, 1.1)


# Bottom plot: Difference (Error)
difference = numerical_results - expected_results
axs[1].plot(delta_s_values, difference, label='Difference (Numerical - Target)', linewidth=1.5, color='blue')
#axs[1].plot(delta_s_values, integration_errors, label='Quad Estimated Error', linewidth=1, linestyle=':', color='gray') # Optional
axs[1].set_xlabel(r'Angle Difference $\Delta s = s_1 - s_2$ (radians)')
axs[1].set_ylabel('Absolute Error')
axs[1].legend()
axs[1].grid(True)
axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Scientific notation for y-axis

# Set x-axis ticks for radians
plt.xticks(np.linspace(0, np.pi, 5), ['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
plt.xlim(0, np.pi)

plt.tight_layout()
plt.show()

# --- Sanity Check: Numerical Normalization ---
norm_integral, norm_err = quad(rho_proposed, 0, 2*np.pi)
print(f"\nNumerical check of normalization for rho_proposed: Integral = {norm_integral:.6f} (Error Est. = {norm_err:.2e})")
print(f"Is normalization close to 1? {np.isclose(norm_integral, 1.0)}")