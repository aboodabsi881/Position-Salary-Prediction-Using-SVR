# Position Salary Prediction Using Support Vector Regression (SVR)

A machine learning project designed to predict a candidate's appropriate salary based on their organizational position level using a Support Vector Regression (SVR) architecture. 

## 📌 Project Description
Accurately mapping corporate job levels to competitive salaries is critical for human resource analytics, recruiting optimization, and compensation benchmarking. Traditional linear regression often fails to capture the true curve of executive salary scaling, which tends to grow exponentially at senior levels. 

This project explores an end-to-end regression pipeline implementing **Support Vector Regression (SVR)**. SVR utilizes a margin-of-tolerance (epsilon) corridor to handle non-linear structures efficiently, making it highly robust for specialized, high-impact numerical predictions.

## 📊 Dataset Specifications
The pipeline utilizes a dataset named `Position_Salaries.csv` with the following characteristics:
* **Total Samples:** 10 discrete position levels (from junior to executive C-suite).
* **Features Used:**
  * `Level`: The independent numerical feature representing rank/seniority (scaled $1$ to $10$).
  * `Salary`: The target continuous numerical value representing annual compensation.
* **Data Split Strategy:** Because the dataset is small and localized, a traditional train-test split was intentionally omitted to maximize the structural density available for curve-fitting.

## 🛠️ Machine Learning Workflow

### 1. Dual-Axis Feature Scaling
Support Vector Regression relies explicitly on distance calculations between data instances. Features with larger raw numerical orders of magnitude (like `Salary`) will skew gradient bounds if left unscaled. 
* To resolve this, two independent `StandardScaler` instances are configured.
* **Feature Transformation ($X$):** Standardized using a dedicated scale matrix.
* **Target Transformation ($y$):** Reshaped to a 2D format, scaled independently, and returned to its natural 1D vector shape via `np.ravel()`.

### 2. SVR Model Architecture
* **Algorithm:** Support Vector Regression (`SVR` via `sklearn.svm`)
* **Kernel Function:** `linear`

### 3. Prediction Pipeline & Inverse Mapping
When inferring values for new feature records, the values must pass through the exact forward and reverse scaling transforms:
1. Transform raw arbitrary features (e.g., intermediate Level `3.5`) using the fitted feature scaler `sc_X`.
2. Generate the scaled predictions using the trained SVR regressor.
3. Apply `.inverse_transform()` via the target scaler `sc_y` to map the result back to human-readable dollar amounts.

**Sample Test Query Output:**
* **Input Rank Level:** `3.5`
* **Predicted Annual Salary:** `~$86,700.55`

## 📈 Visualizations
The project features built-in `matplotlib` plotting sequences. It creates a scatter plot mapping out the position levels against the scaled continuous salary scale using a customized color gradient array (`cmap='viridis'`) to visually emphasize the feature boundaries.

## 🚀 Getting Started

### Prerequisites
Make sure your environment has Python 3 installed alongside the core data science libraries:
```bash
pip install numpy pandas scikit-learn matplotlib
