# Wave Unification Theory (WUT) - Public Code Repository

## Overview

This repository contains Python scripts used to generate figures and perform numerical analyses for publications related to Wave Unification Theory (WUT). WUT is a theoretical physics framework proposing a deterministic, local hidden variable (LHV) approach based on a wave-only ontology to address foundational challenges in physics. The theory aims to provide mechanistic explanations for quantum phenomena and unify physical interactions through the dynamics of an underlying "Rayne field" and its wave structures.

This work is primarily authored by Roger E. Gray, Independent Researcher.

**Primary Foundational Paper (Zenodo - "Paper 1 v3" in development context):**
Gray, R. E. (2024). Wave Unification Theory: A Mechanistic LHV Model for Local, Deterministic Quantum Statistics. *Zenodo*. \url{https://doi.org/10.5281/zenodo.15454660}

## Repository Purpose

The primary purpose of this repository is to provide transparency and enable reproducibility for the numerical simulations and graphical representations presented in WUT publications. Researchers interested in understanding the computational basis of the figures or exploring the models are encouraged to use these scripts.

## Directory Structure

The repository is organized by paper, using the Zenodo DOI as a unique identifier for the most comprehensive versions.

```
wut-public/
|-- requirements.txt                 # Main Python package requirements
|
|-- Paper 1 - WUT Proposal - 15297174/  # Scripts for figures in the initial proposal
|   |-- paper1_figure1.py
|   |-- paper1_figure2.py
|   |-- paper1_figure3.py
|   |-- paper1_figure4.py
|   |-- paper1_figure5.py
|   |-- paper1_figure6.py
|
|-- Paper 2 - Bell Deep Dive - 15327331/ # Scripts for a more focused paper on Bell tests
|   |-- paper2_figure1.py
|   |-- paper2_figure4.py
|
|-- Paper1 v3 -  15454660/             # Scripts for figures in the comprehensive Zenodo paper
|   |-- Figure 1 - Three Polarizer Experiment Visualization/
|   |   |-- figure1.py
|   |-- Figure 2 - Three Polarizer Experiment Intensity/
|   |   |-- figure2.py
|   |-- Figure 3 - Bell Numerical Verification/
|   |   |-- figure3.py
|   |-- Figure 4 - LHV Joint Probability/
|   |   |-- figure4.py
|   |-- Figure 5 - Correlations of Different Variables/
|   |   |-- figure5.py
|   |-- Figure 6 - WUT LHV and QM Correlations/ 
|   |   |-- figure6.py
|   |-- Figure 7 and 8 - Parameter Tuning SSE and a-b Convergence/
|   |   |-- figure7_and_figure8.py
|   |-- Figure 9 - Comparison of WUT and QM CDF/
|   |   |-- figure9_step1_create_npz.py  # Simulates Schrödinger evolution & WUT sampling
|   |   |-- figure9_step2_plot_results.py # Plots comparison for Born rule emergence
|   |-- Figure 10 - Validation of Malus/
|   |   |-- figure10.py


## Getting Started

### Dependencies

The primary Python packages required to run these scripts are generally:
*   `numpy`
*   `scipy`
*   `matplotlib`

A `requirements.txt` file is provided in the root directory. You can install these dependencies using pip:
```bash
pip install -r requirements.txt
```
Some specific scripts or tools might have additional dependencies, which would be noted in their respective subdirectories if applicable.

### Running the Scripts

Each Python script is designed to generate a specific figure or perform a particular analysis. Navigate to the directory containing the script and run it using a Python interpreter:
```bash
python <script_name>.py
```
For multi-part scripts (e.g., `figure9_step1_create_npz.py` and `figure9_step2_plot_results.py`), run them in the indicated order, as the first step typically generates data used by the second.

## Code Description by Paper and Figure

### Paper1 v3 - 15454660 (Zenodo Foundational Paper)

This paper details the comprehensive WUT LHV model and its application to various quantum phenomena.

*   **Figure 1 & 2: Three Polarizer Experiment**
    *   `Figure 1 - Three Polarizer Experiment Visualization/figure1.py`: Generates a conceptual diagram illustrating the difference between a hypothetical 'filtering-only' model and the WUT 'state transformation' model for the three-polarizer experiment (0°, 45°, 90°).
    *   `Figure 2 - Three Polarizer Experiment Intensity/figure2.py`: Plots the predicted relative transmitted intensity after three polarizers as a function of the middle polarizer's angle, comparing the 'filtering-only' model with the WUT 'state transformation' model, which matches experimental observations.

*   **Figure 3 & 4: Bell Test LHV Model**
    *   `Figure 3 - Bell Numerical Verification/figure3.py`: Numerically verifies the core integral of the WUT LHV model for Bell tests, showing that \(\int \rho(\lambda) O(s_1,\lambda) O(s_2,\lambda) d\lambda\) with the proposed \(\rho(\lambda) = \frac{1}{4}|\cos(2\lambda)|\) correctly yields \(\cos(2\Delta s)\).
    *   `Figure 4 - LHV Joint Probability/figure4.py`: Plots the four joint probabilities \(P(++|\Delta s)\), \(P(+-|\Delta s)\), etc., derived from the WUT LHV model as a function of the angle difference \(\Delta s\), demonstrating their match with QM predictions for the singlet state.

*   **Figures 5-8: Continuous Variables & Parameter Tuning (Mechanistic Detector Model)**
    *   `Figure 5 - Correlations of Different Variables/figure5.py`: Plots the statistical correlations for different *continuous* physical variables (e.g., \(q_{final}\), \(W_{bias}\)) derived from the WUT mechanistic detector model (using baseline parameters \(a=1, b=1\)), comparing them to the QM target.
    *   `Figure 6 - WUT LHV and QM Correlations/figure6.py`: (This script seems to generate Figure 3 from the journal article draft). It compares the binary correlation \(E_{reported}\) and the untuned continuous state correlation \(E_{cont}^{(q_{final})}\) with the QM target, and also visualizes the magnitude of the continuous state \(q_{final}(L)\) versus the implicit binary magnitude.
    *   `Figure 7 and 8 - Parameter Tuning SSE and a-b Convergence/figure7_and_figure8.py`: This script performs numerical optimization of the detector potential parameters ($a, b$) to minimize the Sum of Squared Error (SSE) between the continuous correlation \(E_{cont}^{(q_{final})}\) and \(E_{QM}\). It generates plots showing the convergence of ($a-b$) to \(\sqrt{2/3}\) and the decrease of SSE as the parameters increase. Results are saved to `optimization_results/`.

*   **Figure 9: Born Rule Emergence Simulation**
    *   `Figure 9 - Comparison of WUT and QM CDF/figure9_step1_create_npz.py`: Simulates the 1D Time-Dependent Schrödinger Equation for a Gaussian wave packet scattering off a finite barrier (Phase 1). Then, it performs Monte Carlo sampling of detection events based on the WUT hypothesis that detection probability is proportional to the local wave intensity (\(|\psi|^2\)) (Phase 2). Results are saved to an `.npz` file.
    *   `Figure 9 - Comparison of WUT and QM CDF/figure9_step2_plot_results.py`: Loads the data from the `.npz` file generated by `step1` and creates comparison plots (PDFs, residuals, CDFs) between the simulated WUT detection distribution and the QM target, providing computational evidence for the plausible emergence of Born rule statistics.

*   **Figure 10: Malus's Law Validation**
    *   `Figure 10 - Validation of Malus/figure10.py`: Plots and compares the theoretical Malus's Law \(P(\theta) = \cos^2(\theta)\) with the result derived from the WUT LHV model using the hypothesized post-polarizer distribution \(P(\lambda) = \cos(2\lambda)\) and the detection rule \(O(s,\lambda) = \text{sgn}[\cos(2(\lambda-s))]\), confirming their mathematical equivalence.

### Paper 2 - Bell Deep Dive - 15327331

This paper likely focused more specifically on the Bell test LHV model and its mechanistic underpinnings.
*   `paper2_figure1.py`: Generates Figure 1 for this paper, which shows the binary correlation \(E_{reported}\) derived from the mechanistic relaxation model (using state-shift anticorrelation) matching the QM target.
*   `paper2_figure4.py`: Generates Figure 4 for this paper, comparing the continuous correlation \(E_{cont}^{(q_{final})}\) for different WUT detector potential parameter sets (Baseline, User Guess, Found Opt) against the QM target.

### Paper 1 - WUT Proposal - 15297174

This appears to be an earlier foundational proposal. The figures likely illustrate core concepts or initial model validations.
*   `paper1_figure1.py` and `paper1_figure2.py`: These are identical to those in `Paper1 v3` (Zenodo:15454660) for the Three Polarizer Experiment visualization and intensity plot, respectively.
*   `paper1_figure3.py`: This is identical to `figure9_step2_plot_results.py` from `Paper1 v3`, plotting the Born Rule simulation results.
*   `paper1_figure4.py`: This is identical to `figure10.py` from `Paper1 v3`, validating Malus's Law.
*   `paper1_figure5.py`: This is identical to `figure3.py` from `Paper1 v3`, showing the Bell LHV model integral verification.
*   `paper1_figure6.py`: This is identical to `figure4.py` from `Paper1 v3`, plotting the Bell LHV joint probabilities.



## How to Cite

If you use or refer to the code in this repository, please cite the relevant publication(s) for which the code was developed and this GitHub repository.

**For the LHV model for Bell/Malus statistics and related figures (primarily from `Paper1 v3 - 15454660` directory):**
*   Gray, R. E. (2025). Wave Unification Theory: A Mechanistic LHV Model for Local, Deterministic Quantum Statistics. *Zenodo*. \url{https://doi.org/10.5281/zenodo.15454660}
*   Gray, R. E. (2025). WUT-Public: Code for Wave Unification Theory Publications. GitHub Repository. \url{https://github.com/Emerson4642/WUT-Public}




## Disclaimer

Wave Unification Theory (WUT) is an ongoing research project proposing a novel foundational framework for physics. The models and interpretations presented are part of this developing theory and are subject to further refinement, validation, and peer review. The code provided here is for the purpose of reproducing the published or pre-printed results and enabling further exploration by the scientific community.

## License

Please refer to the LICENSE file in this repository (if one is added). If no license file is present, the code is provided under standard copyright with all rights reserved, pending explicit licensing. (It is highly recommended to add an open-source license like MIT or Apache 2.0).
```