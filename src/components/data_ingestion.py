import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.components.generate_data import generate_student_dataset
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("data", "train.csv")
    test_data_path: str = os.path.join("data", "test.csv")
    raw_data_path: str = os.path.join("data", "student_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("Entered the data ingestion method or component")
        try:
            # Check if dataset exists, if not generate it
            if not os.path.exists(self.ingestion_config.raw_data_path):
                print("Dataset not found. Generating synthetic dataset.")
                generate_student_dataset(output_path=self.ingestion_config.raw_data_path)
            
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            print("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            print("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            print("Ingestion of data completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    best_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Model Training Complete! Best R2 Score: {best_score}")
