import dice_ml
from dice_ml import Dice
import pandas as pd
import numpy as np

def run_counterfactuals(model, X_train_scaled, X_test_scaled, y_train, y_test):
    df = X_train_scaled.copy()
    df["target"] = y_train

    data_obj = dice_ml.Data(
        dataframe=df,
        continuous_features=list(X_train_scaled.columns),
        outcome_name="target"
    )

    model_obj = dice_ml.Model(model=model, backend="sklearn")
    exp = Dice(data_obj, model_obj, method="genetic") 
    
    probs = model.predict_proba(X_test_scaled)
    borderline_index = np.argmin(np.abs(probs[:, 1] - 0.5))
    patient = X_test_scaled.iloc[[borderline_index]]

    cf = exp.generate_counterfactuals(
        patient,
        total_CFs=3,
        desired_class="opposite"
    )

    print("Original patient:")
    print(patient.iloc[0])
    print("\nCounterfactual examples:")
    print(cf.cf_examples_list[0].final_cfs_df)
    