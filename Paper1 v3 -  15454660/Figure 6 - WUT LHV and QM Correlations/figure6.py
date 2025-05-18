# ref internal file 'TOP FILE - Bell - Malus LHV/bell3_advanced_success/1d3_Paper2_Figure3.py'

import numpy as np
from scipy.integrate import quad, IntegrationWarning
import matplotlib.pyplot as plt
import warnings

# --- Constants ---
L_crit = 2.0 / (3.0 * np.sqrt(3.0)) # Approx 0.3849
q_crit = 2.0 / np.sqrt(3.0)         # Approx 1.1547

# --- Core Functions ---
# q_final for a=1, b=1 (untuned) - CORRECTED ANALYTICAL
def q_final_untuned_analytical_CORRECTED(L):
    if np.isclose(L, 0.0): return 0.0
    tol = 1e-11; abs_L = np.abs(L)
    if abs_L > L_crit + tol:
        discriminant_term = np.maximum(1e-30, (L / 2.0)**2 - 1.0 / 27.0); sqrt_delta = np.sqrt(discriminant_term)
        term1_base = L / 2.0 + sqrt_delta; term2_base = L / 2.0 - sqrt_delta
        A = np.cbrt(term1_base); B = np.cbrt(term2_base)
        q = A + B
    elif abs_L < L_crit - tol:
        arg_arccos = np.clip(L * np.sqrt(27.0) / 2.0, -1.0, 1.0); phi = np.arccos(arg_arccos)
        if L > 0: q = (2.0 / np.sqrt(3.0)) * np.cos(phi / 3.0) # k=0
        else: q = (2.0 / np.sqrt(3.0)) * np.cos(phi / 3.0 + 2.0 * np.pi / 3.0) # k=1
    else: q = np.sign(L) * q_crit
    if np.isclose(L, 0.0): return 0.0
    return q

# Use the corrected analytical function
def C_direct_qfinal_untuned(L):
    return q_final_untuned_analytical_CORRECTED(L)

# --- Rho and Integrand (remain the same) ---
def rho(lam): return 0.25 * np.abs(np.cos(2.0 * lam))
def integrand_qfinal_state_shift(lam, s1, s2, C_func):
    L1 = np.cos(2.0 * (lam - s1)); L2 = -np.cos(2.0 * (lam - s2))
    C1_val = C_func(L1); C2_val = C_func(L2); rho_val = rho(lam)
    if not (np.isfinite(C1_val) and np.isfinite(C2_val)): return 0.0
    return rho_val * C1_val * C2_val
# --- End Functions ---


# --- Calculate Correlation Data ---
delta_s_values = np.linspace(0, np.pi / 2, 100)
s2 = 0.0
E_cont_qfinal_untuned = []
cusp_points = [np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
integration_limit = 200; integration_epsabs=1e-10; integration_epsrel=1e-10
print("Calculating E_continuous (untuned)...")
# ... (Integration loop) ...
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=IntegrationWarning)
    for i, delta_s in enumerate(delta_s_values):
        s1 = delta_s + s2
        result, error = quad(integrand_qfinal_state_shift, 0, 2 * np.pi,
                             args=(s1, s2, C_direct_qfinal_untuned),
                             limit=integration_limit, points=cusp_points,
                             epsabs=integration_epsabs, epsrel=integration_epsrel)
        E_cont_qfinal_untuned.append(result)
print("Integration complete.")
E_cont_qfinal_untuned = np.array(E_cont_qfinal_untuned)
E_qm_values = -np.cos(2.0 * delta_s_values)
E_reported_values = E_qm_values
# --- End Calculation ---


# --- Create Final Plot with Custom Layout ---
fig = plt.figure(figsize=(10, 8)) # Adjust figsize if needed
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[1, 1]) # Keep height ratio

# --- Top Left Panel (ax1) ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(delta_s_values / np.pi, E_reported_values, label=r'$E_{reported}$ (Binary)', linewidth=2, color='black')
ax1.plot(delta_s_values / np.pi, E_qm_values, label=r'$E_{QM}$', linestyle='--', color='red', linewidth=2)
ax1.set_title('(a) Binary Outcome Correlation (Match)')
ax1.set_xlabel(r'$\Delta s / \pi$')
ax1.set_ylabel('Correlation E')
ax1.grid(True); ax1.legend(); ax1.axhline(0, color='gray', linewidth=0.5)

# --- Top Right Panel (ax2) ---
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.plot(delta_s_values / np.pi, E_cont_qfinal_untuned, label=r'$E_{cont}^{(q_{final})}$ (Untuned)', linewidth=2, color='blue')
ax2.plot(delta_s_values / np.pi, E_qm_values, label=r'$E_{QM}$', linestyle='--', color='red', linewidth=2)
ax2.set_title('(b) Untuned Continuous State Correlation (Mismatch)')
ax2.set_xlabel(r'$\Delta s / \pi$')
ax2.grid(True); ax2.legend(); ax2.axhline(0, color='gray', linewidth=0.5)
# Adjust shared Y axis limit
min_data_cont = np.min(E_cont_qfinal_untuned); max_data_cont = np.max(E_cont_qfinal_untuned)
min_lim = min(min_data_cont, -1.0); max_lim = max(max_data_cont, 1.0)
ax1.set_ylim(min_lim - 0.1, max_lim + 0.1)


# --- Bottom Panel (ax3): Magnitude Comparison Plot (Spanning Full Width) ---
ax3 = fig.add_subplot(gs[1, :]) # Row 1, span all columns ':'
L_plot = np.linspace(-1.1, 1.1, 500)
q_plot = np.array([q_final_untuned_analytical_CORRECTED(L) for L in L_plot])
mag_q = np.abs(q_plot)

ax3.plot(L_plot, mag_q, color='blue', linewidth=2, label=r'$|q_{final}(L)|$ (Untuned Continuous)')
ax3.axhline(1, color='black', linestyle='--', linewidth=1.5, label='$|sgn(L)| = 1$ (Binary Magnitude)')
ax3.fill_between(L_plot, 1, mag_q, where=mag_q >= 1, color='red', alpha=0.3, interpolate=True, label='Excess Magnitude Region')

ax3.set_title('(c) Magnitude Comparison')
ax3.set_xlabel(r'Input Bias $L$')
ax3.set_ylabel('Magnitude of State/Outcome')
ax3.grid(True, linestyle=':')
ax3.legend(fontsize='small')
# *** MODIFIED Y-LIMITS FOR PANEL C ***
max_magnitude = np.max(mag_q)
ax3.set_ylim(0.5, max_magnitude * 1.05) # Start Y at 0.5

# --- Overall Adjustments ---
fig.suptitle('WUT LHV: Binary vs. Untuned Continuous Correlation & Underlying Mechanism', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
# Optional: Adjust vertical space between top and bottom rows if needed
# plt.subplots_adjust(hspace=0.3)
plt.show()

# --- Optional: Save the figure ---
figure_filename = "figure_3_final_layout_magnitude_comparison_zoomedY.png"
# fig.savefig(figure_filename, dpi=300)
# print(f"Figure saved as {figure_filename}")