import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# --- Helper Functions ---

def draw_polarizer(ax, center, angle_deg, radius, label):
    """Draws a polarizer symbol."""
    angle_rad = np.deg2rad(angle_deg)
    # Outer circle
    circ = patches.Circle(center, radius, fill=False, ec='black', lw=1.5)
    ax.add_patch(circ)
    # Line indicating orientation
    x1 = center[0] - radius * np.cos(angle_rad)
    y1 = center[1] - radius * np.sin(angle_rad)
    x2 = center[0] + radius * np.cos(angle_rad)
    y2 = center[1] + radius * np.sin(angle_rad)
    ax.add_line(Line2D([x1, x2], [y1, y2], color='black', lw=1.5))
    # Label
    ax.text(center[0], center[1] + radius * 1.3, f'{label}\n({angle_deg}°)',
            ha='center', va='bottom', fontsize=10)

def draw_state_polar(fig, center, dominant_angle_deg, radius, label, state_type='fixed'):
    """Draws a small polar plot representing the polarization state."""
    ax_polar = fig.add_axes([center[0]-radius, center[1]-radius, 2*radius, 2*radius],
                             projection='polar')
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.spines['polar'].set_visible(False)
    ax_polar.set_rmax(1.1)

    dominant_angle_rad = np.deg2rad(dominant_angle_deg)

    if state_type == 'fixed':
        # Single arrow for fixed polarization state
        ax_polar.arrow(dominant_angle_rad, 0, 0, 1,
                       color='blue', head_width=0.3, head_length=0.3, length_includes_head=True)
        ax_polar.text(0, 0, label, ha='center', va='center', fontsize=8, color='blue')

    elif state_type == 'wut_transformed':
        # Distribution P(lambda) = cos(2*lambda_relative)
        # We plot abs(cos(2*delta_angle)) for visualization over [-pi/4, pi/4] range
        theta = np.linspace(dominant_angle_rad - np.pi/4, dominant_angle_rad + np.pi/4, 100)
        delta_angle = theta - dominant_angle_rad
        # Use abs() or max(0,...) for visual intensity/probability representation
        r = np.abs(np.cos(2 * delta_angle))
        #r = np.maximum(0, np.cos(2 * delta_angle)) # Alternative
        ax_polar.plot(theta, r, color='red')
        ax_polar.fill(theta, r, color='red', alpha=0.3)
        # Indicate center direction
        ax_polar.arrow(dominant_angle_rad, 0, 0, 0.6, # Shorter arrow
                       color='black', head_width=0.2, head_length=0.2,
                       length_includes_head=True, ls='--')
        ax_polar.text(0, 0, label, ha='center', va='center', fontsize=8, color='red')
    ax_polar.set_facecolor('none')

# --- Main Plot Setup ---
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
fig.suptitle("Three Polarizer Experiment (0°, 45°, 90°): Visualization", fontsize=16)

# --- Define Positions ---
y_hypothetical = 5.5
y_wut = 2.5
x_start = 1
x_pol1 = 3
x_state1 = 4.5
x_pol2 = 6.5
x_state2 = 8
x_pol3 = 10
x_final = 11.5
x_end = 13

pol_radius = 0.5
state_radius = 0.05 # Adjusted relative size for add_axes [left, bottom, width, height]
state_disp_radius = 0.4 # Visual radius within the polar plot axes

# --- 1. Hypothetical Model (Filtering Only) ---
ax.text(x_start - 0.5, y_hypothetical + 1, "Model 1: Hypothetical 'Filtering-Only'", fontsize=12, weight='bold')

# Initial Light (Assume polarized at 0° after some implicit first stage)
ax.arrow(x_start, y_hypothetical, x_pol1 - x_start - pol_radius, 0,
         head_width=0.15, head_length=0.3, fc='gray', ec='gray', length_includes_head=True)
ax.text(x_start + (x_pol1-x_start)/2, y_hypothetical + 0.2, "Input Light\n(Intensity $I_0$)", ha='center', fontsize=9)
# State before Pol 1 (implicitly 0°)

# Polarizer 1 (0°)
draw_polarizer(ax, (x_pol1, y_hypothetical), 0, pol_radius, "Pol 1")

# Light after Pol 1
intensity1 = 1.0 # Assume I_0 through first polarizer = 1.0 for simplicity
ax.arrow(x_pol1 + pol_radius, y_hypothetical, x_state1 - (x_pol1 + pol_radius) - state_disp_radius, 0,
         head_width=0.15, head_length=0.3, fc='blue', ec='blue', length_includes_head=True,
         lw= intensity1 * 2) # Line width represents intensity
ax.text(x_pol1 + (x_state1-x_pol1)/2, y_hypothetical + 0.2, f"$I_1 = {intensity1:.2f} I_0$", ha='center', fontsize=9)
# State after Pol 1
draw_state_polar(fig, (x_state1, y_hypothetical), 0, state_disp_radius, "State:\nFixed 0°", state_type='fixed')

# Arrow to Pol 2
ax.arrow(x_state1 + state_disp_radius, y_hypothetical, x_pol2 - (x_state1 + state_disp_radius) - pol_radius, 0,
         head_width=0.15, head_length=0.3, fc='blue', ec='blue', length_includes_head=True,
         lw= intensity1 * 2)

# Polarizer 2 (45°)
draw_polarizer(ax, (x_pol2, y_hypothetical), 45, pol_radius, "Pol 2")

# Light after Pol 2
angle_diff_12 = 45 - 0
intensity2_hyp = intensity1 * (np.cos(np.deg2rad(angle_diff_12))**2)
ax.arrow(x_pol2 + pol_radius, y_hypothetical, x_state2 - (x_pol2 + pol_radius) - state_disp_radius, 0,
         head_width=0.15, head_length=0.3, fc='blue', ec='blue', length_includes_head=True,
         lw= intensity2_hyp * 2)
ax.text(x_pol2 + (x_state2-x_pol2)/2, y_hypothetical + 0.2, f"$I_2 = I_1 \cos^2(45°) $\n$    = {intensity2_hyp:.2f} I_0$", ha='center', fontsize=9)
# State after Pol 2 (Hypothetical - still fixed 0°)
draw_state_polar(fig, (x_state2, y_hypothetical), 0, state_disp_radius, "State:\nStill 0°\n(Hypothetical)", state_type='fixed')

# Arrow to Pol 3
ax.arrow(x_state2 + state_disp_radius, y_hypothetical, x_pol3 - (x_state2 + state_disp_radius) - pol_radius, 0,
         head_width=0.15, head_length=0.3, fc='blue', ec='blue', length_includes_head=True,
         lw= intensity2_hyp * 2)

# Polarizer 3 (90°)
draw_polarizer(ax, (x_pol3, y_hypothetical), 90, pol_radius, "Pol 3")

# Final Light (Hypothetical)
angle_diff_state2_3_hyp = 90 - 0 # Compare Pol 3 to the HYPOTHETICAL state (still 0°)
intensity3_hyp = intensity2_hyp * (np.cos(np.deg2rad(angle_diff_state2_3_hyp))**2)
ax.arrow(x_pol3 + pol_radius, y_hypothetical, x_final - (x_pol3 + pol_radius), 0,
         head_width=0.15, head_length=0.3, fc='gray', ec='gray', length_includes_head=True,
         lw= max(0.1, intensity3_hyp * 2)) # Minimum line width > 0 for visibility
ax.text(x_final + 0.5, y_hypothetical, f"$I_3 = I_2 \cos^2(90°) $\n$    = {intensity3_hyp:.2f} I_0$", ha='left', va='center', fontsize=10, color='red', weight='bold')
ax.text(x_final + 0.5, y_hypothetical - 0.5, "Prediction: Blocked!", ha='left', va='center', fontsize=10, color='red')


# --- 2. WUT Model (State Transformation) ---
ax.text(x_start - 0.5, y_wut + 1, "Model 2: WUT 'State Transformation'", fontsize=12, weight='bold')

# Initial Light (Assume polarized at 0° after some implicit first stage)
ax.arrow(x_start, y_wut, x_pol1 - x_start - pol_radius, 0,
         head_width=0.15, head_length=0.3, fc='gray', ec='gray', length_includes_head=True)
ax.text(x_start + (x_pol1-x_start)/2, y_wut + 0.2, "Input Light\n(Intensity $I_0$)", ha='center', fontsize=9)

# Polarizer 1 (0°)
draw_polarizer(ax, (x_pol1, y_wut), 0, pol_radius, "Pol 1")

# Light after Pol 1
intensity1 = 1.0 # Same initial intensity after Pol 1
ax.arrow(x_pol1 + pol_radius, y_wut, x_state1 - (x_pol1 + pol_radius) - state_disp_radius, 0,
         head_width=0.15, head_length=0.3, fc='red', ec='red', length_includes_head=True,
         lw = intensity1 * 2)
ax.text(x_pol1 + (x_state1-x_pol1)/2, y_wut + 0.2, f"$I_1 = {intensity1:.2f} I_0$", ha='center', fontsize=9)
# State after Pol 1 (WUT - Transformed)
draw_state_polar(fig, (x_state1, y_wut), 0, state_disp_radius, "State:\nTransformed\n(Rel. 0°)", state_type='wut_transformed')

# Arrow to Pol 2
ax.arrow(x_state1 + state_disp_radius, y_wut, x_pol2 - (x_state1 + state_disp_radius) - pol_radius, 0,
         head_width=0.15, head_length=0.3, fc='red', ec='red', length_includes_head=True,
         lw = intensity1 * 2)

# Polarizer 2 (45°)
draw_polarizer(ax, (x_pol2, y_wut), 45, pol_radius, "Pol 2")

# Light after Pol 2
angle_diff_12 = 45 - 0 # Same calculation for intensity reduction
intensity2_wut = intensity1 * (np.cos(np.deg2rad(angle_diff_12))**2)
ax.arrow(x_pol2 + pol_radius, y_wut, x_state2 - (x_pol2 + pol_radius) - state_disp_radius, 0,
         head_width=0.15, head_length=0.3, fc='red', ec='red', length_includes_head=True,
         lw= intensity2_wut * 2)
ax.text(x_pol2 + (x_state2-x_pol2)/2, y_wut + 0.2, f"$I_2 = I_1 \cos^2(45°) $\n$    = {intensity2_wut:.2f} I_0$", ha='center', fontsize=9)
# State after Pol 2 (WUT - Transformed AGAIN)
draw_state_polar(fig, (x_state2, y_wut), 45, state_disp_radius, "State:\nTransformed!\n(Rel. 45°)", state_type='wut_transformed')

# Arrow to Pol 3
ax.arrow(x_state2 + state_disp_radius, y_wut, x_pol3 - (x_state2 + state_disp_radius) - pol_radius, 0,
         head_width=0.15, head_length=0.3, fc='red', ec='red', length_includes_head=True,
         lw= intensity2_wut * 2)

# Polarizer 3 (90°)
draw_polarizer(ax, (x_pol3, y_wut), 90, pol_radius, "Pol 3")

# Final Light (WUT)
# Intensity reduction depends on angle between Pol 3 (90°) and the axis of Pol 2 (45°)
angle_diff_23 = 90 - 45
intensity3_wut = intensity2_wut * (np.cos(np.deg2rad(angle_diff_23))**2)
ax.arrow(x_pol3 + pol_radius, y_wut, x_final - (x_pol3 + pol_radius), 0,
         head_width=0.15, head_length=0.3, fc='red', ec='red', length_includes_head=True,
         lw= max(0.1, intensity3_wut * 2))
ax.text(x_final + 0.5, y_wut, f"$I_3 = I_2 \cos^2(90°-45°) $\n$    = 0.5 \cdot 0.5 I_0$\n$    = {intensity3_wut:.2f} I_0$", ha='left', va='center', fontsize=10, color='green', weight='bold')
ax.text(x_final + 0.5, y_wut - 0.5, "Prediction: Passes!", ha='left', va='center', fontsize=10, color='green')


# --- Explanatory Text ---
ax.text(0.5, 0.5,
        "Key Difference:\n"
        "Model 1 assumes the state after Pol 1 (0°) remains fixed.\n"
        "  -> Interaction with Pol 3 (90°) uses Δθ = 90°-0° = 90°. Result: Blocked.\n"
        "Model 2 (WUT) assumes the state is transformed by Pol 2 (45°).\n"
        "  -> Interaction with Pol 3 (90°) uses Δθ = 90°-45° = 45°. Result: Passes.\n"
        "Experiment matches Model 2.",
        ha='left', va='bottom', fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.8))


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.show()