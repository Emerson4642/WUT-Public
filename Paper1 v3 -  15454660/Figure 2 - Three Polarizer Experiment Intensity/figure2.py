# Reference internal research file 'TOP FILE - Bell - Malus LHV/three_polarizer/paper1_figure2.py'

import numpy as np
import matplotlib.pyplot as plt

# --- Constants and Setup ---
THETA1_DEG = 0      # Angle of the first polarizer (degrees)
THETA3_DEG = 90      # Angle of the third polarizer (degrees)
I0_RELATIVE = 1.0  # Assume intensity after Pol 1 is 1.0 for relative comparison

# --- Angle Range for Second Polarizer ---
theta2_deg = np.linspace(0, 180, 360) # Vary theta2 from 0 to 180 degrees
theta2_rad = np.deg2rad(theta2_deg)

# --- Intensity Calculations ---

# Model 1: Hypothetical 'Filtering-Only'
intensity3_model1 = np.zeros_like(theta2_deg) # Result is always 0

# Model 2: WUT 'State Transformation'
intensity3_model2 = I0_RELATIVE * np.cos(theta2_rad)**2 * np.sin(theta2_rad)**2

# --- Specific Example Point (theta2 = 45 degrees) ---
theta2_example_deg = 45
theta2_example_rad = np.deg2rad(theta2_example_deg)
intensity_example = I0_RELATIVE * np.cos(theta2_example_rad)**2 * np.sin(theta2_example_rad)**2

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot Model 1 Prediction
plt.plot(theta2_deg, intensity3_model1, 'b--', lw=2,
         label='Model 1 Prediction (Filtering-Only): $I_3/I_0 = 0$')

# Plot Model 2 Prediction (Matches Experiment)
plt.plot(theta2_deg, intensity3_model2, 'r-', lw=2,
         label='Model 2 Prediction (WUT State Transformation): $I_3/I_0 = \\cos^2(\\theta_2)\\sin^2(\\theta_2)$')

# Plot the specific example point
plt.plot(theta2_example_deg, intensity_example, 'go', markersize=8,
         label=f'Example: $\\theta_2={theta2_example_deg}° \\Rightarrow I_3/I_0={intensity_example:.2f}$')
plt.vlines(theta2_example_deg, 0, intensity_example, colors='gray', linestyles='dotted')
plt.hlines(intensity_example, 0, theta2_example_deg, colors='gray', linestyles='dotted')

# --- Labels and Formatting ---
plt.xlabel('Angle of Second Polarizer, $\\theta_2$ (degrees)')
plt.ylabel('Relative Transmitted Intensity, $I_3 / I_0$')
plt.title('Three Polarizer Experiment: Final Intensity vs. Middle Polarizer Angle\n(Pol 1 @ {}°, Pol 3 @ {}°)'.format(THETA1_DEG, THETA3_DEG))
plt.ylim(-0.02, 0.3) # Set y-limit slightly below 0 for visibility of the blue line
plt.xlim(0, 180)
plt.xticks(np.arange(0, 181, 30))
plt.grid(True, linestyle=':')
plt.legend(loc='upper right') # Keep legend in upper right

# Add explanatory text - MOVED TO AVOID OVERLAP
plt.text(95, 0.05, "Experimental observations match the\n'State Transformation' model (Red Curve)",
         fontsize=10, ha='left', va='bottom', 
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

plt.tight_layout()
plt.show()