# NPZDO Model

This repository contains the implementation and analysis of the NPZDO (Nutrient-Phytoplankton-Zooplankton-Detritus-Oxygen) model.

## Repository Structure

* **`report.pdf`**: The final project report
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

### 2. Nutrient Variation Analysis
To observe how the system reacts to nutrient changes, run the **`# Low/high nutrient input`** subsection specifically, rather than executing the entire output block.

### 3. Sensitivity Analysis
To conduct a sensitivity analysis:
1. Navigate to the **`# Parameters`** section.
2. Manually adjust the constant values (e.g., growth rates, mortality).
3. Re-run the **`# Output`** section to generate the updated results.
