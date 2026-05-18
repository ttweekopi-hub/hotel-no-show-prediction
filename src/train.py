import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
import lightgbm as lgb
from src.logger import get_logger

logger = get_logger("Train")

def train_and_select_model(
    X_train: np.ndarray, 
    X_test: np.ndarray, 
    y_train: np.ndarray, 
    y_test: np.ndarray,
    models_dir: str = "models"
):
    """
    Trains multiple candidate models: LightGBM, Random Forest, and Logistic Regression.
    Evaluates their performance on the test split, compares their ROC-AUC and F1-Scores,
    and automatically serializes the best performing model.
    """
    logger.info("Initializing model training and evaluation phase...")
    
    # Initialize candidate models with optimized parameters
    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150, 
            learning_rate=0.05, 
            random_state=42, 
            n_jobs=-1,
            verbosity=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=12, 
            random_state=42, 
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, 
            C=1.0, 
            random_state=42, 
            n_jobs=-1
        )
    }
    
    best_roc_auc = 0.0
    best_model_name = None
    best_model_obj = None
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name} Classifier...")
        try:
            model.fit(X_train, y_train)
            
            # Predict and evaluate on test split
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            roc_auc = roc_auc_score(y_test, probs)
            
            results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            }
            
            logger.info(f"{name} Metrics: F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, Accuracy={acc:.4f}")
            
            # Track best model based on ROC-AUC
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_model_name = name
                best_model_obj = model
                
        except Exception as e:
            logger.error(f"Failed to train {name}: {str(e)}")
            
    # Print out comparison summary table in console
    logger.info("=" * 60)
    logger.info(f"{'Model Name':<20} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Accuracy':<10}")
    logger.info("-" * 60)
    for name, metrics in results.items():
        logger.info(f"{name:<20} | {metrics['F1-Score']:<10.4f} | {metrics['ROC-AUC']:<10.4f} | {metrics['Accuracy']:<10.4f}")
    logger.info("=" * 60)
    
    logger.info(f"🏆 Best model based on ROC-AUC: {best_model_name} with ROC-AUC = {best_roc_auc:.4f}")
    
    # Save the absolute best model object
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    with open(best_model_path, "wb") as f:
        pickle.dump(best_model_obj, f)
        
    logger.info(f"Fitted best model ({best_model_name}) serialized and saved to {best_model_path}")
    
    return best_model_name, results
