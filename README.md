# Multi-Dimensional Marketing Mix Modeling (MMM) Engine

An R&D econometrics framework written in Python 3 that automates the optimization of multi-channel advertising decay effects (**Adstock constants**) using global non-linear optimization algorithms. 

## 🧠 Technical Highlights

Unlike naive grid-search or single-variable approaches, this engine optimizes the entire system of advertising factors simultaneously to prevent greedy convergence.

* **Global Non-Linear Optimization**: Leverages SciPy's `differential_evolution` algorithm to discover optimal global $\lambda$ decay parameters across multiple dimensions.
* **Overfitting Mitigation (Custom Multi-Objective Loss)**: Features a robust objective function minimizing the Euclidean norm (`np.linalg.norm`) of concatenated metric errors ($R^2$, SAE, MAPE) simultaneously across both Train and Test splits.
* **Production-Ready Pipeline**: Includes automated missing-value imputation, numerical schema alignment, sequential time-series mapping, and direct visual analytics mapping.

## 🛠️ Tech Stack

* **Core**: Python 3, NumPy, Pandas
* **Modeling & Optimization**: Scikit-Learn (Linear Regression), SciPy (Differential Evolution)
* **Visualization**: Matplotlib

## 🚀 Quick Start & Architecture

The execution pipeline functions in three seamless steps:

1. **Data Preprocessing**: Automatically handles `NaN` values, sets a chronological tracking index, and excludes non-modeling targets.
2. **Mathematical Optimization**: Computes exponential Adstock transformations iteratively:
   $$Carry_t = X_t + \lambda \cdot Carry_{t-1}$$
   Then passes hyperparameters to the differential evolution engine.
3. **Validation & Diagnostics**: Trains the final model, outputs localized performance metrics, and plots actual vs. predicted values divided precisely by the Train/Test boundary.

