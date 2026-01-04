import os,sys, yaml
from src.utils.exception_handler import MyException
from src.utils.logger import logging
from src.entity import DataIngestionArtifact, DataTransformationArtifact
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dotenv import load_dotenv
from src.utils.common import read_yaml_file, write_yaml_file
import numpy as np
import joblib

load_dotenv()

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.target_column = os.getenv('TARGET_COLUMN')
            self.data_schema = read_yaml_file(os.getenv("SCHEMA_FILE_PATH"))
            self.artifacts_dir = os.getenv("DATA_ROOT_DIR")
        except Exception as e:
            raise MyException(e, sys)

    def create_preprocessing_pipeline(self) -> ColumnTransformer:
        try:
            num_features = self.data_schema['num_columns']
            mm_features = self.data_schema['mm_columns']
            categorical_features = self.data_schema['categorical_columns']
            drop_columns = self.data_schema['drop_columns']
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('mm', MinMaxScaler(), mm_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
                    ('drop', 'drop', drop_columns)
                ],
                remainder='passthrough'
            )
            return preprocessor
        except Exception as e:
            logging.error(f"Error in creating preprocessing pipeline: {e}")
            raise MyException(e, sys)

    def run(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation process...")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            preprocessor = self.create_preprocessing_pipeline()

            X_train = train_df.drop(self.target_column, axis=1)
            y_train = train_df[self.target_column]
            X_test = test_df.drop(self.target_column, axis=1)
            y_test = test_df[self.target_column]

            logging.info("Applying preprocessing transformations...")
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Preprocessing transformations applied successfully.")

            logging.info("Applying SMOTEENN to handle class imbalance...")
            smote_enn = SMOTEENN(random_state=42)
            X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train_transformed, y_train)
            logging.info("SMOTEENN applied successfully.")

            train_array = np.c_[X_train_resampled, y_train_resampled]
            test_array = np.c_[X_test_transformed, y_test]
            
            logging.info("Saving transformed data and preprocessing object...")
            transformed_dir = os.path.join(self.artifacts_dir, self.data_ingestion_artifact.date_dir, os.getenv("DATA_TRANSFORMATION_DIR_NAME"))
            os.makedirs(transformed_dir, exist_ok=True)
            transformed_data_dir = os.path.join(transformed_dir, os.getenv("DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR"))
            os.makedirs(transformed_data_dir, exist_ok=True)
            transformed_train_file_path = os.path.join(transformed_data_dir, os.getenv("TRANSFORMED_TRAIN_FILE_NAME"))
            transformed_test_file_path = os.path.join(transformed_data_dir, os.getenv("TRANSFORMED_TEST_FILE_NAME"))

            transformed_object = os.path.join(transformed_dir, os.getenv("DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR"))
            os.makedirs(transformed_object, exist_ok=True)
            preprocessed_object_file_path = os.path.join(transformed_object, os.getenv("PREPROCESSED_OBJECT_FILE_NAME"))

            np.save(transformed_train_file_path, train_array)
            np.save(transformed_test_file_path, test_array)
            with open(preprocessed_object_file_path, 'wb') as f:
                joblib.dump(preprocessor, f)
            data = {
                'transformed_columns': preprocessor.get_feature_names_out().tolist()
            }
            write_yaml_file(file_path=os.getenv('TRANSFORMED_COLUMNS_ORDERING_FILE_NAME'), content=data )
            logging.info("Transformed data and preprocessing object saved successfully.")
            logging.info("Data transformation process completed successfully.")
            return DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                preprocessed_object_file_path=preprocessed_object_file_path
            )
        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise MyException(e, sys)