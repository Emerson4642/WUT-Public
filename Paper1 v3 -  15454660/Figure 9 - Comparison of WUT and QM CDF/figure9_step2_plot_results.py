#ref internal file 'Schrodinger_Born/phase3_paper_1_figure_3.py'

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest
from scipy.integrate import cumulative_trapezoid # Needed for theoretical CDF
from scipy.interpolate import interp1d # Needed for theoretical CDF

def run_phase3_revised(results_file="phase2_results.npz"):
    """Loads Phase 2 data and performs revised Phase 3 analysis."""
    print("-" * 40)
    print("--- Phase 3 (Revised): Analysis and Comparison ---")

    # Load data from Phase 2
    try:
        data = np.load(results_file)
        P_qm_final = data['P_qm']
        sampled_x_positions = data['x_samples']
        x_grid = data['x_grid']
        dx = data['dx']
        N_samples = data['N_samples'].item() # Ensure N_samples is int
        t_final = data['t_final'].item()     # Ensure t_final is float
        print(f"Loaded data from {results_file} (N={N_samples})")
    except FileNotFoundError:
        print(f"ERROR: Results file '{results_file}' not found.")
        return None, None # Indicate failure
    except Exception as e:
        print(f"ERROR: Loading data failed - {e}")
        return None, None # Indicate failure

    # --- Task 3a & 3b: Create and Normalize Histogram ---
    num_bins_list = [150, 100, 50] # Test multiple bin numbers for Chi2

    # --- Calculate Theoretical CDF (needed for KS test and plotting) ---
    cdf_theoretical = cumulative_trapezoid(P_qm_final, dx=dx, initial=0)
    cdf_theoretical = cdf_theoretical / cdf_theoretical[-1] # Ensure normalization

    # --- Task 3c: Visual Comparison Plots ---
    print("Generating comparison plots (Density, Residuals, CDF)...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # 1. Density Plot
    ax = axes[0]
    # Plot the target QM probability density
    ax.plot(x_grid, P_qm_final, 'r-', lw=2, label=f'$P_{{QM}}(x) = |\psi(x, t={t_final:.1f})|^2$ (Target)') # Added .1f for t_final
    # Plot the normalized histogram of simulated WUT detections (use a fixed reasonable bin number for plot)
    # hist_density_for_plot will be the DENSITY values if density=True
    hist_density_for_plot, bin_edges_plot, _ = ax.hist(
             sampled_x_positions, bins=150, range=(x_grid[0], x_grid[-1]),
             density=True, alpha=0.7, label=f'WUT Samples Histogram (N={N_samples})')
    bin_centers_plot = (bin_edges_plot[:-1] + bin_edges_plot[1:]) / 2

    ax.set_title('Comparison of WUT Simulation Samples to QM Probability Density')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Probability Density')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(x_grid[0], x_grid[-1])

    # 2. Residual Plot
    ax = axes[1]
    # Interpolate P_qm_final onto bin centers for comparison
    P_qm_interp_for_residuals = np.interp(bin_centers_plot, x_grid, P_qm_final)
    
    # CORRECTED: Use the density values directly from the histogram plotted above
    # hist_density_for_plot IS ALREADY THE DENSITY
    residuals = hist_density_for_plot - P_qm_interp_for_residuals 

    ax.plot(bin_centers_plot, residuals, 'bo-', markersize=3, label='Residuals (Histogram - Target PDF)')
    ax.axhline(0, color='r', linestyle='--')
    ax.set_title('Residuals between Histogram and Target PDF')
    ax.set_xlabel('Position x (Bin Center)')
    ax.set_ylabel('Density Difference')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(x_grid[0], x_grid[-1])

    # 3. CDF Plot
    ax = axes[2]
    # Calculate Empirical CDF (ECDF)
    ecdf_x = np.sort(sampled_x_positions)
    ecdf_y = np.arange(1, N_samples + 1) / N_samples
    
    # Plot Theoretical CDF (solid red) first (so it's underneath)
    ax.plot(x_grid, cdf_theoretical, 'r-', lw=2, alpha=0.9, label='Theoretical CDF (Target)') 
    
    # Plot Empirical CDF (dashed dodgerblue) second (so it's on top)
    ax.plot(ecdf_x, ecdf_y, color='dodgerblue', linestyle='--', lw=1.5, label='Empirical CDF (Samples)') 
    
    ax.set_title('Comparison of Cumulative Distribution Functions (CDF)')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Cumulative Probability')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(x_grid[0], x_grid[-1])


    plt.tight_layout()
    plot_filename = "phase3_comparison_revised_plots.png"
    plt.savefig(plot_filename)
    print(f"Comparison plots saved as {plot_filename}")
    # plt.show()
    plt.close(fig) # Close figure window

    # --- Task 3d: Quantitative Comparison ---

    # Adjustment 1: Chi-squared Test with different bins
    print("-" * 20)
    print("Performing Chi-squared tests with varying bins...")
    for num_bins in num_bins_list:
        print(f"--- Chi-squared Test ({num_bins} bins) ---")
        hist_counts, bin_edges = np.histogram(sampled_x_positions, bins=num_bins, range=(x_grid[0], x_grid[-1]))
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # Not strictly needed for this calculation
        bin_width = bin_edges[1] - bin_edges[0]

        expected_counts = np.zeros_like(hist_counts, dtype=float)
        for i in range(num_bins):
            # Define the integration sub-grid for the current bin
            x_bin_subgrid = x_grid[(x_grid >= bin_edges[i]) & (x_grid < bin_edges[i+1])]
            P_qm_bin_subgrid = P_qm_final[(x_grid >= bin_edges[i]) & (x_grid < bin_edges[i+1])]

            if len(x_bin_subgrid) > 1:
                 prob_in_bin = np.trapz(P_qm_bin_subgrid, x_bin_subgrid)
            elif len(x_bin_subgrid) == 1:
                 # If only one grid point falls in the bin, approximate using that point's density * bin_width
                 # This is a rough approximation, better if dx << bin_width
                 prob_in_bin = P_qm_bin_subgrid[0] * bin_width 
            else:
                 # No x_grid points in this bin. This can happen if bin_width is smaller than dx.
                 # A more robust way if dx is coarse: interpolate P_qm_final at bin edges & center, then integrate.
                 # For now, assume if no grid points, probability is very small or zero.
                 # Or, ensure num_bins is chosen such that bin_width > dx.
                 # A simple interpolation at bin center might be an alternative:
                 # P_qm_interp_center = np.interp((bin_edges[i] + bin_edges[i+1])/2, x_grid, P_qm_final)
                 # prob_in_bin = P_qm_interp_center * bin_width
                 prob_in_bin = 0.0 
            expected_counts[i] = N_samples * prob_in_bin
        
        # Filter bins: Observed > 0 AND Expected >= 5 (common rule)
        valid_bins_mask = (hist_counts > 0) & (expected_counts >= 5)
        num_valid_bins = np.sum(valid_bins_mask)

        if num_valid_bins < num_bins * 0.5 or num_valid_bins < 5: # Heuristic check
            print(f"  WARNING: Only {num_valid_bins}/{num_bins} bins have Obs>0 and Exp>=5.")
            print("  Chi-squared test result might be unreliable. Consider fewer bins or more robust expected_counts.")
            if num_valid_bins < 2: # Need at least 2 bins for dof > 0
                 print("  Skipping Chi-squared test for this binning.")
                 continue

        chi2_stat = np.sum(((hist_counts[valid_bins_mask] - expected_counts[valid_bins_mask])**2) / expected_counts[valid_bins_mask])
        dof = num_valid_bins - 1 # Dof based on number of bins *used* in calculation
        p_value = chi2.sf(chi2_stat, dof)

        print(f"  Using {num_valid_bins} bins with Exp >= 5.")
        print(f"  Chi-squared Statistic (\u03C7\u00B2) = {chi2_stat:.4f}")
        print(f"  Degrees of Freedom (dof) = {dof}")
        print(f"  P-value = {p_value:.4g}")
        alpha = 0.05
        if p_value > alpha:
             print(f"  Conclusion: Consistent (p > {alpha}).")
        else:
             print(f"  Conclusion: INconsistent (p <= {alpha}).")
    print("-" * 20)

    # Adjustment 2: Kolmogorov-Smirnov Test
    print("Performing Kolmogorov-Smirnov test...")
    # Need a function representing the theoretical CDF for kstest
    # We can use interpolation on the calculated theoretical CDF
    cdf_theoretical_func = interp1d(x_grid, cdf_theoretical,
                                    kind='linear', bounds_error=False,
                                    fill_value=(0.0, 1.0))

    ks_statistic, ks_p_value = kstest(sampled_x_positions, cdf_theoretical_func)

    print(f"  KS Statistic = {ks_statistic:.6f}")
    print(f"  P-value = {ks_p_value:.4g}")
    alpha = 0.05
    if ks_p_value > alpha:
        print(f"  Conclusion: The sample distribution is statistically consistent with the target distribution (p > {alpha}).")
    else:
        print(f"  Conclusion: The sample distribution is statistically INconsistent with the target distribution (p <= {alpha}).")

    print("-" * 40)
    print("--- Phase 3 (Revised) Complete ---")
    return P_qm_final, sampled_x_positions # Return results if needed

# --- How to Run ---
if __name__ == "__main__":
    # Assuming 'phase2_results.npz' exists from running Phase 1 & 2 script
    # You would typically call your Phase 1 and Phase 2 functions here first, e.g.:
    # from phase1_tdse_solver import run_phase1
    # from phase2_wut_sampling import run_phase2
    #
    # params = {...} # Define your simulation parameters
    # psi_final, x_grid_p1, dx_p1, t_final_p1 = run_phase1(**params)
    # if psi_final is not None:
    #     run_phase2(psi_final, x_grid_p1, dx_p1, N_samples=500000, t_final=t_final_p1)
    #     run_phase3_revised(results_file="phase2_results.npz")
    # else:
    #     print("Phase 1 failed, skipping subsequent phases.")

    # For standalone testing of Phase 3, ensure "phase2_results.npz" is present.
    run_phase3_revised()