"""
This module runs the inference pipeline to make predictions on new hotel reservations.

I built this module to act as our live prediction engine. It loads the fitted preprocessor
and the trained best model, transforms new customer reservation entries, and outputs whether
they are predicted to be a no-show alongside their absolute probability score.
"""

import os
import pickle
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger("Predict")

def make_predictions(df: pd.DataFrame, models_dir: str = "models") -> pd.DataFrame:
    """Runs the full inference pipeline on raw new reservation records.

    I have created this interface to cleanly score fresh customer bookings:
      - Automatically checks that preprocessor and model pickles are fully loaded.
      - Standardizes first_time binary guest variables.
      - Transforms categorical and numeric features strictly using the saved parameters.
      - Returns class predictions and absolute probability scores for business analysts.

    Args:
        df: A pd.DataFrame containing one or more raw reservation booking rows.
        models_dir: The directory where the preprocessor and model are loaded from.
            Defaults to 'models'.

    Returns:
        A pd.DataFrame matching the input rows, appended with 'predicted_probability',
        'predicted_no_show' (0/1), and 'predicted_no_show_label' (No/Yes) columns.

    Raises:
        FileNotFoundError: If the best model or preprocessor pickle assets are missing.
    """
    logger.info("Initializing inference pipeline...")
    
    # Paths for model and preprocessor
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    
    if not os.path.exists(best_model_path) or not os.path.exists(preprocessor_path):
        err_msg = f"Inference assets missing in {models_dir}. Ensure training runs successfully first."
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
        
    # Load serializations
    logger.info("Loading preprocessor and model...")
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)
    with open(best_model_path, "rb") as f:
        best_model = pickle.load(f)
        
    logger.info("Executing transformations on input features...")
    # Clean binary categories
    df_features = df.copy()
    if "first_time" in df_features.columns:
        df_features["first_time"] = df_features["first_time"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        
    categorical_cols = ["branch", "country", "room"]
    numerical_cols = ["price_sgd", "stay_duration", "lead_time_months"]
    binary_cols = ["first_time"]
    
    feature_cols = categorical_cols + numerical_cols + binary_cols
    X_inference = df_features[feature_cols]
    
    # Process features
    X_processed = preprocessor.transform(X_inference)
    
    logger.info("Running model predictions...")
    probs = best_model.predict_proba(X_processed)[:, 1]
    preds = best_model.predict(X_processed)
    
    # Output final predictions appended to copy
    output_df = df.copy()
    output_df["predicted_probability"] = probs
    output_df["predicted_no_show"] = preds
    output_df["predicted_no_show_label"] = output_df["predicted_no_show"].apply(lambda x: "Yes" if x == 1 else "No")
    
    logger.info(f"Inference completed successfully. Generated predictions for {len(output_df)} rows.")
    return output_df
