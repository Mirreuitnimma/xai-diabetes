# XAI for Medical Risk Assessment
**Course:** Explainable AI 2025/2026 | **Author:** Mirre van der Poel

This project applies two XAI methods to a diabetes risk prediction task:
- **DiCE** — Counterfactual Explanations
- **Incremental Bayesian Network** — Step-by-step probabilistic explanations

---

## Requirements
- Python 3.9 or higher
- pip

---

## Installation

### 1. Download the project and navigate to the folder
```
cd path/to/explainable AI project
```

### 2. Create and activate a virtual environment
```
python -m venv venv
```
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 3. Install required packages
```
pip install scikit-learn pandas numpy matplotlib dice-ml pgmpy shap
```

---

## Usage

### Run the full project
```
python main.py
```

This will automatically:
1. Load and preprocess the diabetes dataset
2. Train a Logistic Regression model
3. Run DiCE counterfactual explanations
4. Run the Incremental Bayesian Network explanation

### Expected output
**In the terminal:**
```
Model accuracy: 0.73

=== COUNTERFACTUAL EXPLANATIONS ===
Original patient: ...
Counterfactual examples: ...

=== INCREMENTAL BAYESIAN NETWORK ===
Incremental probability updates:
  Prior (no evidence): 0.33
  + bmi = low: 0.22
  + bp = medium: 0.18
  + s1 = high: 0.36
  + s5 = medium: 0.24
  + age = high: 0.50
```

**A pop-up figure will appear** showing the incremental BN probability updates as a bar chart. Close the plot window to allow the program to finish.

---

## Dataset
This project uses the `load_diabetes` dataset from scikit-learn — no separate download needed, it loads automatically. It contains 442 patients with 10 features (age, sex, BMI, blood pressure, and 6 blood serum measurements). The continuous target is converted to binary classification using the 70th percentile as threshold, representing high vs. low diabetes risk.

---

## Reproducibility
All results are based on a fixed random seed (`random_state=42`). Running `main.py` will always produce the same output.
