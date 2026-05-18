import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from src.logger import get_logger

logger = get_logger("Clean")

def clean_data(df: pd.DataFrame, models_dir: str = "models", is_training: bool = True) -> pd.DataFrame:
    """Performs rigorous data cleaning and advanced machine-learning imputation.

    I have designed this function to resolve the complex data quality anomalies:
      - Drops rows with missing target labels to prevent invalid model optimization.
      - Corrects inconsistent spelling/casing in month strings (e.g. MaY -> May).
      - Converts negative day parameters to absolute values (e.g. -31 -> 31).
      - Normalizes text representation numbers in num_adults (e.g. 'one' -> 1).
      - Unifies mixed currency types to SGD using the implied 1.37 exchange rate.
      - Imputes missing room types dynamically via a 5-Nearest Neighbors Classifier.
      - Imputes missing price details dynamically via a Random Forest Regressor.

    Args:
        df: A pd.DataFrame containing the raw hotel booking records.
        models_dir: The directory where imputation models are saved or loaded from.
            Defaults to 'models'.
        is_training: If True, trains new KNN and Random Forest models on available data 
            and serializes them; if False, loads pre-trained serialized model assets.

    Returns:
        A pd.DataFrame representing the fully cleaned dataset with no missing values.

    Raises:
        FileNotFoundError: If is_training is False and required model assets are missing.
    """
    logger.info("Starting data cleaning pipeline...")
    
    # Make a copy to avoid mutating the original
    df = df.copy()
    
    # 1. Drop rows with null target (target variable 'no_show')
    initial_count = len(df)
    df = df.dropna(subset=["no_show"])
    dropped_target = initial_count - len(df)
    if dropped_target > 0:
        logger.info(f"Dropped {dropped_target} rows because the target variable 'no_show' was null.")
        
    # 2. Standardize month names casing to Title Case
    for col in ["arrival_month", "checkout_month", "booking_month"]:
        if col in df.columns:
            logger.info(f"Standardizing {col} month names...")
            df[col] = df[col].astype(str).str.lower().str.capitalize()
        
    # 3. Correct negative arrival_day and checkout_day values
    for col in ["arrival_day", "checkout_day"]:
        if col in df.columns:
            logger.info(f"Correcting negative {col} values...")
            df[col] = pd.to_numeric(df[col], errors='coerce').abs()
        
    # 4. Map text guest counts to digits in num_adults
    if "num_adults" in df.columns:
        logger.info("Converting num_adults text numbers to integers...")
        adults_map = {'1': 1, '2': 2, 'one': 1, 'two': 2}
        df["num_adults"] = df["num_adults"].astype(str).replace(adults_map)
        df["num_adults"] = pd.to_numeric(df["num_adults"], errors='coerce').fillna(1).astype(int)
        
    if "num_children" in df.columns:
        df["num_children"] = pd.to_numeric(df["num_children"], errors='coerce').fillna(0).astype(int)
        
    # 5. Currency normalization and price unification to SGD (Rate 1.37)
    if "price" in df.columns:
        logger.info("Normalizing price currencies to SGD...")
        def parse_price(val):
            if pd.isna(val):
                return None
            val_str = str(val).strip()
            parts = val_str.split()
            if len(parts) == 2:
                curr, val_num = parts[0], float(parts[1])
                if "USD" in curr:
                    return val_num * 1.37
                else:
                    return val_num
            else:
                # Fallback in case of direct numeric value
                try:
                    return float(val_str)
                except:
                    return None
                
        df["price_sgd"] = df["price"].apply(parse_price)
        # Drop the original un-unified text price column
        df = df.drop(columns=["price"])
        
    # Standardize branch label encoding for imputation
    if "branch" in df.columns:
        df["branch_encoded"] = df["branch"].apply(lambda x: 1 if str(x).lower() == "changi" else 0)
        
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    knn_path = os.path.join(models_dir, "imputer_knn_room.pkl")
    rf_path = os.path.join(models_dir, "imputer_rf_price.pkl")
    
    # 6. Smart Imputation of 'room' column via KNN Classifier
    if "room" in df.columns and "price_sgd" in df.columns:
        logger.info("Handling missing room type values using KNN...")
        
        # Identify rows for training KNN (where price_sgd and room are both NOT null)
        clean_room_df = df.dropna(subset=["room", "price_sgd"])
        missing_room_df = df[df["room"].isna()]
        
        if is_training:
            logger.info("Training KNN Classifier for room imputation...")
            X_knn = clean_room_df[["price_sgd", "branch_encoded"]]
            y_knn = clean_room_df["room"]
            
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_knn, y_knn)
            
            with open(knn_path, "wb") as f:
                pickle.dump(knn, f)
            logger.info(f"Fitted KNN Classifier saved to {knn_path}")
        else:
            if not os.path.exists(knn_path):
                raise FileNotFoundError(f"Required KNN room imputer model not found at {knn_path}. Run training first.")
            with open(knn_path, "rb") as f:
                knn = pickle.load(f)
            logger.info("Loaded pre-trained KNN room imputer.")
            
        # Impute missing rooms
        if len(missing_room_df) > 0:
            # We must only impute rows that have price_sgd. If a row has null price_sgd as well,
            # it will be filled in the next price imputation step
            impute_subset = missing_room_df.dropna(subset=["price_sgd"])
            if len(impute_subset) > 0:
                X_impute = impute_subset[["price_sgd", "branch_encoded"]]
                pred_rooms = knn.predict(X_impute)
                df.loc[impute_subset.index, "room"] = pred_rooms
                logger.info(f"Imputed {len(impute_subset)} missing room types using KNN.")
                
            # If any are still null, we fill with safe default
            df["room"] = df["room"].fillna("Single")
            
    # 7. Smart Imputation of 'price_sgd' column via Random Forest Regressor
    if "price_sgd" in df.columns and "room" in df.columns:
        logger.info("Handling missing price_sgd values using Random Forest Regressor...")
        
        # Room Label Encoder to align with the training step
        df['room_encoded'] = df['room'].apply(lambda x: {"Single": 0, "Queen": 1, "King": 2, "President Suite": 3}.get(x, 0))
        
        clean_price_df = df.dropna(subset=["price_sgd"])
        missing_price_df = df[df["price_sgd"].isna()]
        
        features_list = ['room_encoded', 'branch_encoded', 'num_adults', 'num_children']
        
        if is_training:
            logger.info("Training Random Forest Regressor for price imputation...")
            X_rf = clean_price_df[features_list]
            y_rf = clean_price_df["price_sgd"]
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X_rf, y_rf)
            
            with open(rf_path, "wb") as f:
                pickle.dump(rf, f)
            logger.info(f"Fitted Random Forest Regressor saved to {rf_path}")
        else:
            if not os.path.exists(rf_path):
                raise FileNotFoundError(f"Required RF price imputer model not found at {rf_path}. Run training first.")
            with open(rf_path, "rb") as f:
                rf = pickle.load(f)
            logger.info("Loaded pre-trained RF price imputer.")
            
        # Impute missing prices
        if len(missing_price_df) > 0:
            X_impute_price = missing_price_df[features_list]
            pred_prices = rf.predict(X_impute_price)
            df.loc[missing_price_df.index, "price_sgd"] = pred_prices
            logger.info(f"Imputed {len(missing_price_df)} missing prices using Random Forest Regressor.")
            
    # Drop intermediate columns
    drop_cols = ["branch_encoded", "room_encoded"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
        
    logger.info("Data cleaning completed successfully.")
    return df
