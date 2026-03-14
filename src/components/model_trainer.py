import os
import sys
import json
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("models", "model.joblib")
    model_report_file_path = os.path.join("models", "model_report.json")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            print("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            
            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    'criterion': ['squared_error', 'absolute_error', 'poisson'],
                    'max_depth': [3, 5, 8, 10]
                },
                "Random Forest": {
                    'n_estimators': [32, 64, 128, 256],
                    'max_depth': [5, 8, 10, None]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'subsample': [0.7, 0.8, 0.9],
                    'n_estimators': [32, 64, 128]
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05],
                    'n_estimators': [32, 64, 128],
                    'max_depth': [3, 5, 8]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }
            }

            print("Evaluating all 7 machine learning models")
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            # Save evaluation report to JSON file
            os.makedirs(os.path.dirname(self.model_trainer_config.model_report_file_path), exist_ok=True)
            with open(self.model_trainer_config.model_report_file_path, "w") as f:
                json.dump(model_report, f, indent=4)
            print(f"Model report saved to {self.model_trainer_config.model_report_file_path}")

            # Get best model score and name
            best_model_name = ""
            best_model_score = -1.0
            
            for model_name, metrics in model_report.items():
                if metrics["r2_score"] > best_model_score:
                    best_model_score = metrics["r2_score"]
                    best_model_name = model_name
                    
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score >= 0.6")
            
            print(f"Best found model: {best_model_name} with R2 score: {best_model_score}")

            # Fit best model on the entire training data again to finalize
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Save background data for SHAP explanations (first 100 samples)
            bg_data_path = os.path.join("models", "background_data.joblib")
            save_object(
                file_path=bg_data_path,
                obj=X_train[:100]
            )

            return best_model_score
            
        except Exception as e:
            raise CustomException(e, sys)