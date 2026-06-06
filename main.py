import numpy as np
import pandas as pd
import counterfactuals
import bayesian_network

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y_continuous = data.target

threshold = np.percentile(y_continuous, 70)
y = (y_continuous >= threshold).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("Model accuracy:", model.score(X_test, y_test))

patient = X_test.iloc[[0]]

print("\n=== COUNTERFACTUAL EXPLANATIONS ===")
counterfactuals.run_counterfactuals(model, X_train_scaled, X_test_scaled, y_train, y_test)

print("\n=== INCREMENTAL BAYESIAN NETWORK ===")
bayesian_network.run_incremental_bn(X_train_scaled, X_test_scaled, y_train, y_test)
