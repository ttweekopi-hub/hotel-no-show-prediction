"""
This module preprocesses our features and splits the dataset for training.

I designed this module to handle scaling and encoding safely without any data leakage:
  - Maps binary fields (like no_show and first_time) to integer representations.
  - Splits the data into a stratified 80/20 train/test split to preserve label proportions.
  - Scales numerical features using RobustScaler to protect against price outliers.
  - One-hot encodes categorical branches, countries, and rooms using OneHotEncoder.
  - Serializes the fitted preprocessing transformer so we can reuse it during prediction.
"""

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from src.logger import get_logger

logger = get_logger("Preprocess")

def preprocess_data(df: pd.DataFrame, models_dir: str = "models", is_training: bool = True):
    """Executes target mapping, stratified splitting, scaling, and one-hot encoding.

    I have engineered this stage to completely eliminate data leakage:
      - Maps target labels (no_show) to binary integers.
      - Splits the data into stratified 80% training and 20% validation sets.
      - Scales numerical features (price, stay duration, lead time) using RobustScaler
        to defend the model against unexpected pricing outliers.
      - Encodes categorical values (branch, country, room) using OneHotEncoder.
      - Fits the preprocessors strictly on the training set and transforms the test set.

    Args:
        df: A pd.DataFrame containing the featured hotel booking records.
        models_dir: The directory where the fitted preprocessor is saved or loaded from.
            Defaults to 'models'.
        is_training: If True, fits and serializes a new preprocessing pipeline; 
            if False, loads a pre-trained serialized preprocessing pipeline.

    Returns:
        If is_training is True:
            A tuple of (X_train_processed, X_test_processed, y_train, y_test, feature_names)
            containing the processed matrices, raw targets, and engineered column names.
        If is_training is False:
            A tuple of (X_inference_processed, None, None, None, None) for inference.

    Raises:
        FileNotFoundError: If is_training is False and no preprocessor asset is found.
    """
    logger.info("Initializing preprocessing and splitting stage...")
    df = df.copy()
    
    # 1. Map target variable 'no_show' to binary integer
    if "no_show" in df.columns:
        logger.info("Mapping target variable 'no_show' to binary integer...")
        def parse_binary(x):
            val_str = str(x).strip().lower()
            if val_str in ["1", "1.0", "yes", "true", "y"]:
                return 1
            return 0
        df["no_show"] = df["no_show"].apply(parse_binary)
        
    # 2. Map binary features (first_time) to integer (Yes=1, No=0)
    if "first_time" in df.columns:
        logger.info("Mapping guest loyalty status ('first_time') to binary integer...")
        df["first_time"] = df["first_time"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
        
    # Define exact columns to preserve in modeling
    categorical_cols = ["branch", "country", "room"]
    numerical_cols = ["price_sgd", "stay_duration", "lead_time_months"]
    binary_cols = ["first_time"]
    
    feature_cols = categorical_cols + numerical_cols + binary_cols
    
    if is_training:
        X = df[feature_cols]
        y = df["no_show"]
        
        # Perform 80/20 train-test split, stratifying on target to maintain class ratios
        logger.info("Performing stratified 80/20 train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Split completed. Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows.")
        
        # 3. Create ColumnTransformer preprocessing pipeline
        logger.info("Creating ColumnTransformer preprocessor...")
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ("num", RobustScaler(), numerical_cols)
            ],
            remainder="passthrough"  # Passes binary columns (like first_time) straight through
        )
        
        # Fit preprocessor on X_train only and transform
        logger.info("Fitting and transforming preprocessing pipeline on training features only...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Capture and log feature names for full transparency
        try:
            # Reconstruct feature names from encoder
            cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols).tolist()
            feature_names = cat_features + numerical_cols + binary_cols
            logger.info(f"Fitted preprocessing pipeline. Total features: {len(feature_names)}")
        except Exception as e:
            logger.warning(f"Could not extract feature names: {e}")
            feature_names = None
            
        # Serialize fitted preprocessor
        os.makedirs(models_dir, exist_ok=True)
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        with open(preprocessor_path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Fitted preprocessor pipeline saved to {preprocessor_path}")
        
        return X_train_processed, X_test_processed, y_train.values, y_test.values, feature_names
        
    else:
        # Inference mode: Load existing preprocessor
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f" Fails to load preprocessor at {preprocessor_path}. Run training first.")
            
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        logger.info("Loaded pre-trained preprocessor pipeline.")
        
        X_inference = df[feature_cols]
        X_inference_processed = preprocessor.transform(X_inference)
        return X_inference_processed, None, None, None, None
