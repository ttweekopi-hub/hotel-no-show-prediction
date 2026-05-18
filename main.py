import sys
from src.logger import get_logger
from src.ingest import ingest_data
from src.clean import clean_data
from src.features import generate_features
from src.preprocess import preprocess_data
from src.train import train_and_select_model
from src.predict import make_predictions

logger = get_logger("Orchestrator")

def main():
    """Main orchestration engine executing the end-to-end ML pipeline.

    Coordinates all pipeline components:
      1. Ingests raw data from SQLite noshow.db.
      2. Cleans columns and imputes rooms/prices without data leakage.
      3. Engineers duration and advance booking features.
      4. Splits dataset (80/20 stratified) and encodes/scales features.
      5. Trains, compares, tunes, and serializes the best predictive model.
      6. Verifies model inference pipeline on sample inputs.
    """
    logger.info("==================================================")
    logger.info("STARTING HOTEL NO-SHOW MACHINE LEARNING PIPELINE")
    logger.info("==================================================")
    
    try:
        # Step 1: Ingest Data
        logger.info("[Step 1/6] Starting Ingestion Phase...")
        raw_df = ingest_data("data/noshow.db")
        
        # Step 2: Clean & Impute Data (is_training = True to fit KNN & RF imputers)
        logger.info("[Step 2/6] Starting Data Cleaning and Imputation Phase...")
        cleaned_df = clean_data(raw_df, is_training=True)
        
        # Step 3: Feature Engineering
        logger.info("[Step 3/6] Starting Feature Engineering Phase...")
        featured_df = generate_features(cleaned_df)
        
        # Step 4: Preprocessing & Data Splitting (is_training = True to fit standardizer & OHE)
        logger.info("[Step 4/6] Starting Preprocessing and Data Splitting Phase...")
        X_train, X_test, y_train, y_test, features = preprocess_data(featured_df, is_training=True)
        
        # Step 5: Model Training, Tuning, & Serialization
        logger.info("[Step 5/6] Starting Model Training Phase...")
        best_model_name, results = train_and_select_model(X_train, X_test, y_train, y_test)
        
        # Performance Threshold Check for Quality Gate
        # ROC-AUC is our primary metric since it measures the model's ranking ability,
        # which is crucial for hoteliers to rank and contact high-risk no-show reservations.
        best_roc_auc = results[best_model_name]["ROC-AUC"]
        logger.info(f"Model Performance Verification: Primary Metric (ROC-AUC) = {best_roc_auc:.4f}")
        
        MIN_ROC_AUC_THRESHOLD = 0.53
        if best_roc_auc < MIN_ROC_AUC_THRESHOLD:
            err_msg = (
                f"QUALITY GATE FAILURE: Best model's ROC-AUC ({best_roc_auc:.4f}) is below "
                f"the required minimum threshold of {MIN_ROC_AUC_THRESHOLD:.2f}! "
                f"Model predictive power is insufficient to capture no-shows."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
            
        logger.info(f"✅ Quality Gate Passed: Best model's ROC-AUC ({best_roc_auc:.4f}) exceeds baseline of {MIN_ROC_AUC_THRESHOLD:.2f}.")
        
        # Step 6: Inference Test Verification
        logger.info("[Step 6/6] Verifying Inference Pipeline on Sample Cleaned Bookings...")
        # Verify predict.py by sending in a sample subset of featured data
        sample_subset = featured_df.head(100)
        predictions_df = make_predictions(sample_subset)
        
        logger.info("==================================================")
        logger.info("SUCCESS: PIPELINE EXECUTED AND MODEL VERIFIED!")
        logger.info(f"Fitted Best Model: {best_model_name}")
        logger.info("All pipeline logs appended to: logs/pipeline.log")
        logger.info("==================================================")
        
    except Exception as e:
        logger.critical(f"FATAL ERROR: Pipeline execution halted. Details: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
