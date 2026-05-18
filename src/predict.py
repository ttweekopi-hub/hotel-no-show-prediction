import os
import pickle
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger("Predict")

def make_predictions(df: pd.DataFrame, models_dir: str = "models") -> pd.DataFrame:
    """
    Runs the inference pipeline on raw new dataframe bookings:
      1. Loads preprocessor and model.
      2. Makes predictions.
      3. Appends predicted_probability and predicted_no_show columns.
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
