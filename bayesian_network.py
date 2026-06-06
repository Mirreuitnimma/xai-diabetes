import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

def run_incremental_bn(X_train, X_test, y_train, y_test):

    def discretize(df):
        df_discrete = pd.DataFrame()
        for col in df.columns:
            df_discrete[col] = pd.cut(
                df[col],
                bins=3,
                labels=["low", "medium", "high"]
            )
        return df_discrete

    X_train_disc = discretize(X_train)
    X_test_disc = discretize(X_test)

    train_df = X_train_disc.copy()
    train_df["high_risk"] = y_train

    train_df = train_df.astype(str)

    structure = [
        ("bmi", "high_risk"),
        ("bp", "high_risk"),
        ("s1", "high_risk"),
        ("s5", "high_risk"),
        ("age", "high_risk"),
    ]

    model = DiscreteBayesianNetwork(structure)
    model.fit(train_df)

    inference = VariableElimination(model)
    patient = X_test_disc.iloc[0]
    incremental_features = ["bmi", "bp", "s1", "s5", "age"]

    prior = inference.query(["high_risk"])
    prior_prob = float(prior.values[1])  

    probabilities = [prior_prob]
    labels = ["Prior (no evidence)"]

    evidence_so_far = {}
    for feature in incremental_features:
        evidence_so_far[feature] = str(patient[feature])
        result = inference.query(
            ["high_risk"],
            evidence=evidence_so_far
        )
        prob = float(result.values[1])
        probabilities.append(prob)
        labels.append(f"+ {feature} = {patient[feature]}")

    plt.figure(figsize=(10, 6))
    colors = ["red" if p > 0.5 else "blue" for p in probabilities]
    plt.bar(labels, probabilities, color=colors, alpha=0.7)
    plt.axhline(y=0.5, color="black", linestyle="--", label="Decision boundary")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("P(high risk)")
    plt.title("Incremental Bayesian Network Explanation")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.legend()
    plt.show()

    print("\nIncremental probability updates:")
    for label, prob in zip(labels, probabilities):
        print(f"  {label}: {prob:.3f}")
        