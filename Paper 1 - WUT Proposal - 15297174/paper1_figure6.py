import numpy as np
import matplotlib.pyplot as plt

# 1. Define the range of angle differences (Delta s)
delta_s_rad = np.linspace(0, np.pi, 200) # Angle difference in radians

# 2. Calculate the common trigonometric term
cos_2delta_s = np.cos(2 * delta_s_rad)

# 3. Calculate the four joint probabilities using the derived formulas from Appendix C.5
prob_pp = 0.25 * (1 - cos_2delta_s)
prob_pm = 0.25 * (1 + cos_2delta_s)
prob_mp = 0.25 * (1 + cos_2delta_s) # Same as P(+-)
prob_mm = 0.25 * (1 - cos_2delta_s) # Same as P(++)

# 4. Sanity Check
total_prob = prob_pp + prob_pm + prob_mp + prob_mm
print(f"Plotting Joint Probabilities derived from LHV model:")
print(f"Sum of probabilities close to 1? {np.allclose(total_prob, 1.0)}")

# 5. Create the plot
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(10, 6))

# --- REVERTED TO ORIGINAL LINE STYLES ---
# Plot the calculated probabilities with clean labels and distinct styles
ax.plot(delta_s_rad, prob_pp, label=r'$P(++|\Delta s) = \frac{1}{4}[1 - \cos(2\Delta s)]$', linewidth=2, color='green', linestyle='-') # Solid line for P(++)
ax.plot(delta_s_rad, prob_pm, label=r'$P(+-|\Delta s) = \frac{1}{4}[1 + \cos(2\Delta s)]$', linewidth=2, color='red', linestyle='-') # Solid line for P(+-)
ax.plot(delta_s_rad, prob_mp, label=r'$P(-+|\Delta s) = \frac{1}{4}[1 + \cos(2\Delta s)]$', linewidth=2, color='blue', linestyle='--') # Dashed line for P(-+)
ax.plot(delta_s_rad, prob_mm, label=r'$P(--|\Delta s) = \frac{1}{4}[1 - \cos(2\Delta s)]$', linewidth=2, color='magenta', linestyle=':') # Dotted line for P(--)
# --- END REVERTED LINE STYLES ---

# Customize the plot
ax.set_xlabel(r'Angle Difference $\Delta s = s_1 - s_2$ (radians)')
ax.set_ylabel('Joint Probability')
ax.set_title("Joint Probabilities Derived from WUT LHV Model vs. Angle Difference")

# --- ADJUSTED Y-AXIS LIMIT ---
ax.set_ylim(-0.05, 1.05) # Raised upper Y limit
# --- END ADJUSTED Y-AXIS LIMIT ---

ax.set_xticks(np.linspace(0, np.pi, 5))
ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- ADJUSTED TEXT BOX POSITION ---
# Place text box near the top-right corner, higher up
ax.text(np.pi * 0.98, 1.0, # x, y coordinates (y near top)
        'Note: Derived LHV joint probabilities\nexactly match QM predictions\n for the singlet state.',
        horizontalalignment='right', # Align text box edge to the right
        verticalalignment='top',  # Align text box edge to the top
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
# --- END ADJUSTED TEXT BOX POSITION ---

# --- ADJUSTED LEGEND POSITION ---
ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 0.98)) # Place legend just below title
# --- END ADJUSTED LEGEND POSITION ---


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout slightly to prevent title overlap if needed
# Save the figure - Use the filename needed for LaTeX
plot_filename = 'bell_joint_probs_final_v2.png' # Use a new name again
plt.savefig(plot_filename)
print(f"Plot saved as {plot_filename}")

plt.show() # Display the plot