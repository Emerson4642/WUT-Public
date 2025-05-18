# ref internal file 'Schrodinger_Born/phase2_wut_version.py'

# Purpose: Simulate 1D Schrödinger equation (Phase 1) and then
#          perform Monte Carlo sampling based on the result (Phase 2)
#          as part of the WUT Simulation Study.

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

# --- Phase 1 Core Functions ---

def calculate_potential(x_grid, V0, W_barrier):
    """Calculates the finite square barrier potential."""
    V = np.zeros_like(x_grid)
    V[np.abs(x_grid) <= W_barrier / 2.0] = V0
    return V

def calculate_initial_psi(x_grid, x0, sigma_x, k0):
    """Calculates the initial normalized Gaussian wave packet."""
    dx = x_grid[1] - x_grid[0]
    psi0_unnormalized = np.exp(-(x_grid - x0)**2 / (4 * sigma_x**2) + 1j * k0 * x_grid)
    norm_sq = np.sum(np.abs(psi0_unnormalized)**2) * dx
    psi0 = psi0_unnormalized / np.sqrt(norm_sq)
    return psi0

def evolve_split_step(psi_initial, V, x_grid, dt, Nt, hbar=1.0, m=1.0, print_interval=100):
    """Evolves the wavefunction using the Symmetric Split-Step Fourier method."""
    print(f"Starting time evolution: Nt = {Nt} steps, dt = {dt:.3f}...")
    Nx = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    k_grid = 2 * np.pi * fftfreq(Nx, d=dx)
    potential_phase_half = np.exp(-0.5j * V * dt / hbar)
    kinetic_phase = np.exp(-1j * hbar * (k_grid**2) * dt / (2 * m))
    psi_t = psi_initial.copy()
    start_time = time.time()
    for i in range(Nt):
        psi_t = potential_phase_half * psi_t
        psi_k = fft(psi_t)
        psi_k = kinetic_phase * psi_k
        psi_t = ifft(psi_k)
        psi_t = potential_phase_half * psi_t
        if print_interval > 0 and (i + 1) % print_interval == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i+1}/{Nt} completed. Time elapsed: {elapsed:.2f}s")
    end_time = time.time()
    print(f"Time evolution finished. Total computational time: {end_time - start_time:.2f}s")
    return psi_t

# --- Phase 2 Core Functions ---

def calculate_cdf(pdf, dx):
    """Calculates the Cumulative Distribution Function."""
    cdf = cumulative_trapezoid(pdf, dx=dx, initial=0)
    if not np.isclose(cdf[-1], 1.0):
         cdf = cdf / cdf[-1] # Ensure CDF ends exactly at 1.0
    return cdf

def create_inverse_cdf_interpolator(x_grid, cdf):
    """Creates an interpolation function to invert the CDF."""
    unique_cdf_values, unique_indices = np.unique(cdf, return_index=True)
    unique_x = x_grid[unique_indices]
    if not np.isclose(unique_cdf_values[0], 0.0):
        unique_x = np.insert(unique_x, 0, x_grid[0])
        unique_cdf_values = np.insert(unique_cdf_values, 0, 0.0)
    if not np.isclose(unique_cdf_values[-1], 1.0):
        unique_x = np.append(unique_x, x_grid[-1])
        unique_cdf_values = np.append(unique_cdf_values, 1.0)
    inv_cdf_interp = interp1d(unique_cdf_values, unique_x,
                              kind='linear',
                              bounds_error=False,
                              fill_value=(x_grid[0], x_grid[-1]))
    return inv_cdf_interp

def sample_positions_inverse_transform(inv_cdf_interpolator, N_samples):
    """Generates samples using the inverse transform method."""
    uniform_samples = np.random.rand(N_samples)
    sampled_x = inv_cdf_interpolator(uniform_samples)
    return sampled_x

def run_phase2(psi_final, x_grid, N_samples):
    """Performs the Phase 2 Monte Carlo sampling."""
    print("-" * 40)
    print("--- Phase 2: WUT Monte Carlo Sampling ---")
    dx = x_grid[1] - x_grid[0]

    # Task 2a: Calculate Target PDF
    print("Calculating final probability density P(x)...")
    P_qm_final = np.abs(psi_final)**2

    # Task 2b: Normalize PDF Check
    norm_pdf = np.sum(P_qm_final) * dx
    print(f"Numerical check of PDF normalization: {norm_pdf:.10f}")
    if not np.isclose(norm_pdf, 1.0, rtol=1e-6, atol=1e-8):
        print("  WARNING: PDF normalization differs significantly from 1!")
        # P_qm_final = P_qm_final / norm_pdf # Optionally renormalize

    # Task 2c: Implement Sampling (Inverse Transform)
    print("Calculating CDF...")
    cdf_final = calculate_cdf(P_qm_final, dx)
    print("Creating inverse CDF interpolator...")
    inv_cdf_interp = create_inverse_cdf_interpolator(x_grid, cdf_final)

    # Task 2d: Generate Samples
    print(f"Generating N = {N_samples} samples using Inverse Transform Sampling...")
    start_sample_time = time.time()
    sampled_x_positions = sample_positions_inverse_transform(inv_cdf_interp, N_samples)
    end_sample_time = time.time()
    print(f"Sampling finished. Time elapsed: {end_sample_time - start_sample_time:.2f}s")
    print("-" * 40)
    print("--- Phase 2 Complete ---")

    # Return necessary results for Phase 3
    return P_qm_final, sampled_x_positions


# --- Main Execution Block ---
if __name__ == "__main__":
    # === Phase 1 Execution ===
    print("--- WUT Simulation Study - Phase 1: Schrödinger Evolution (Re-run) ---")
    # Define Parameters
    xmin, xmax = -250.0, 250.0      # Increased spatial domain
    Nx = 2**12                      # Increased resolution (4096)
    dx = (xmax - xmin) / Nx
    x_grid = np.linspace(xmin, xmax, Nx, endpoint=False)
    dt = 0.05
    t_final = 40.0
    Nt = int(t_final / dt)
    x0 = -50.0
    sigma_x = 5.0
    k0 = 5.0
    E0 = k0**2 / 2.0
    V0 = 10.0
    W_barrier = 2.0
    hbar = 1.0
    m = 1.0
    print("Parameters set (Increased Domain).")
    print(f"Grid: x in [{xmin:.1f}, {xmax:.1f}], Nx = {Nx}, dx = {dx:.4f}")
    print(f"Time: t_final = {t_final:.1f}, dt = {dt:.3f}, Nt = {Nt}")
    # Calculate Initial State and Potential
    V = calculate_potential(x_grid, V0, W_barrier)
    psi_initial = calculate_initial_psi(x_grid, x0, sigma_x, k0)
    # Perform Time Evolution
    psi_final = evolve_split_step(
        psi_initial, V, x_grid, dt, Nt,
        hbar=hbar, m=m,
        print_interval=Nt // 10
    )
    # Validation
    norm_initial = np.sum(np.abs(psi_initial)**2) * dx
    norm_final = np.sum(np.abs(psi_final)**2) * dx
    print("Phase 1 Validation:")
    print(f"  Initial Norm = {norm_initial:.10f}")
    print(f"  Final Norm   = {norm_final:.10f}")
    if np.isclose(norm_final, 1.0, rtol=1e-6, atol=1e-8):
        print("  Phase 1 Norm Check: PASSED")
    else:
        print("  Phase 1 Norm Check: FAILED")
    # Plotting (Optional - can comment out if not needed for Phase 2 run)
    prob_initial = np.abs(psi_initial)**2
    prob_final = np.abs(psi_final)**2
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(x_grid, prob_initial, label=r'$|\psi(x, t=0)|^2$')
    potential_scale = max(np.max(prob_initial), 0.01) / V0 if V0 > 0 else 1
    plt.plot(x_grid, V * potential_scale, 'r--', label=f'Potential V(x) (scaled)')
    plt.title('Initial State (t=0)')
    plt.xlabel('Position x'); plt.ylabel(r'$|\psi|^2$ / Pot (scaled)'); plt.legend(); plt.grid(True); plt.ylim(bottom=-0.01)
    plt.subplot(2, 1, 2)
    plt.plot(x_grid, prob_final, label=f'$|\psi(x, t={t_final})|^2$')
    plt.plot(x_grid, V * potential_scale, 'r--', label=f'Potential V(x) (scaled)')
    plt.title(f'Final State (t={t_final})')
    plt.xlabel('Position x'); plt.ylabel(r'$|\psi|^2$ / Pot (scaled)'); plt.legend(); plt.grid(True); plt.ylim(bottom=-0.01)
    plt.tight_layout()
    plot_filename = "schrodinger_phase1_rerun_results.png"
    plt.savefig(plot_filename)
    print(f"Phase 1 Plot saved as {plot_filename}")
    # plt.close() # Close plot window if shown automatically

    print("-" * 40)
    print("--- Phase 1 Complete. Proceeding to Phase 2 ---")
    print("-" * 40)

    # === Phase 2 Execution ===
    N_samples = 500000 # Number of WUT detection events to simulate
    P_qm_final, sampled_x_positions = run_phase2(psi_final, x_grid, N_samples)

    # --- Optional: Save Phase 2 results ---
    results_filename = "phase2_results.npz"
    np.savez(results_filename,
             P_qm=P_qm_final,
             x_samples=sampled_x_positions,
             x_grid=x_grid,
             dx=dx,
             N_samples=N_samples,
             t_final=t_final)
    print(f"Phase 2 results saved to {results_filename}")
    print("-" * 40)

    # --- End of Script ---
    print("--- Phase 1 and Phase 2 Scripts Execution Complete ---")
    print("Ready for Phase 3 Analysis using data from phase2_results.npz")