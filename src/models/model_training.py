import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name: str = "credit_risk_prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_high_risk') -> Tuple:
        """
        Prepare data for modeling by splitting into features and target.
        """
        logger.info("Preparing data for modeling")
        X = df.drop(columns=[target_col, 'CustomerId', 'Cluster'])
        y = df[target_col]
        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series):
        """
        Train multiple models with hyperparameter tuning and track experiments.
        """
        models = {
            'logistic_regression': {
                'model': LogisticRegression(),
                'params': {
                    'C': np.logspace(-4, 4, 20),
                    'penalty': ['l1', 'l2']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        best_models = {}
        
        for model_name, config in models.items():
            logger.info(f"Training {model_name}")
            
            with mlflow.start_run(run_name=model_name):
                # Create pipeline
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', config['model'])
                ])

                # Grid search
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid={f'model__{k}': v for k, v in config['params'].items()},
                    cv=5,
                    scoring='roc_auc',
                    n_jobs=-1
                )

                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred)
                }

                # Log everything to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best_model, "model")

                logger.info(f"{model_name} metrics: {metrics}")
                
                best_models[model_name] = {
                    'model': best_model,
                    'metrics': metrics
                }

        return best_models

    def register_best_model(self, models: dict):
        """
        Register the best performing model in MLflow Model Registry.
        """
        best_model_name = None
        best_roc_auc = 0
        
        # Find model with best ROC-AUC
        for model_name, info in models.items():
            if info['metrics']['roc_auc'] > best_roc_auc:
                best_roc_auc = info['metrics']['roc_auc']
                best_model_name = model_name

        if best_model_name:
            logger.info(f"Registering best model: {best_model_name}")
            mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model",
                "credit_risk_model"
            )
            return best_model_name
        else:
            logger.warning("No models found to register")
            return None
