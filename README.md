# NPZDO Model

This repository contains the implementation and analysis of the NPZDO (Nutrient-Phytoplankton-Zooplankton-Detritus-Oxygen) model.

## Repository Structure

* **`25314___Computational_Marine_Ecological_Modelling___Report_1.pdf`**: The final project report
* **`Instructions_Report1.pdf`**: The original guidelines for the project
* **`NPZDO.py`**: The Python implementation
* **`Figures/`**: A directory containing all visualizations used in the report

---

## How to Use the Code

The code is primarily function-based. To replicate specific results from the report, follow the steps below.

### 1. Reproducing Report Results
Locate the `# Output` section in the script. You can adapt the model behavior by setting the following boolean flags to `True` or `False`:

| Parameter | Description |
| :--- | :--- |
| `use_phyto_light_damping` | Enables/disables light attenuation by phytoplankton |
| `use_seasonal_light` | Switches between constant and seasonal solar forcing |
| `use_seasonal_diffusivity` | Toggles seasonal changes in vertical mixing |

Running the `# Output` section will generate:

* Time-depth convergence plots  
* Final vertical profiles of state variables  
* Limiting resource diagnostics

### 2. Nutrient Variation Analysis
To observe how the system reacts to nutrient changes, run the **`# Low/high nutrient input`** subsection specifically, rather than executing the entire output block.

This analysis compares:

* **Baseline conditions** (`N_bottom = 10 mmol N m⁻³`)
* **High nutrient input** (`N_bottom = 100 mmol N m⁻³`)

The model then plots the final vertical profiles of: Nutrients, Phytoplankton, Zooplankton, Detritus, Oxygen and Chlorophyll-a.

### 3. Sensitivity Analysis
To observe how the system reacts to a change of parameter, run the **`# Sensitivity analysis`** subsection specifically, rather than executing the entire output block.

The model includes a **built-in sensitivity analysis function** that allows testing the influence of a single parameter on the final ecosystem state by plotting both final profiles together (with the default parameter value and the modified value).
