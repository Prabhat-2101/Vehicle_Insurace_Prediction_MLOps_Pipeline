import os, sys
import pandas as pd
from dotenv import load_dotenv
from src.utils.logger import logging 
from src.utils.exception_handler import MyException
from src.entity import DataIngestionArtifact, DataValidationArtifact
import yaml
from src.utils.common import read_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact):
        """
        Initializes the DataValidation class using DataIngestionArtifact.
        """
        self.artifacts_dir = os.getenv("DATA_ROOT_DIR")
        self.date_dir = data_ingestion_artifact.date_dir
        self.train_path = data_ingestion_artifact.train_file_path
        self.test_path = data_ingestion_artifact.test_file_path
        self.data_validation_dir = os.path.join(
            self.artifacts_dir,
            self.date_dir,
            os.getenv("DATA_VALIDATION_DIR_NAME")
        )
        self.report_file_path = os.path.join(
            self.data_validation_dir,
            os.getenv("DATA_VALIDATION_REPORT_FILE_NAME")
        )
        logging.info("DataValidation class initialized successfully.")

    def check_column_count(self,df: pd.DataFrame, expected_column_count: int) -> bool:
        """
        Checks if the DataFrame has the expected number of columns.
        """
        actual_column_count = len(df.columns)
        if actual_column_count == expected_column_count:
            logging.info("Column count validation passed.")
            return True
        else:
            logging.warning(f"Column count validation failed. Expected {expected_column_count}, got {actual_column_count}.")
            return False

    def allowed_columns(self,df: pd.DataFrame, allowed_columns: list) -> bool:
        """
        Checks if the DataFrame contains only allowed columns.
        """
        df_columns = set(df.columns)
        allowed_columns_set = set(allowed_columns)
        if df_columns == allowed_columns_set:
            logging.info("Allowed columns validation passed.")
            return True
        else:
            logging.warning("Allowed columns validation failed.")
            missing = allowed_columns_set - df_columns
            extra = df_columns - allowed_columns_set
            logging.error(f"Missing columns: {missing}")
            logging.error(f"Extra columns: {extra}")
            return False

    def run(self):
        try: 
            debug_message = ""
            validation_status = False
            columns = read_yaml_file(os.getenv("SCHEMA_FILE_PATH"))['columns']
            allowed = [list(col.keys())[0] for col in columns]
            
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            
            logging.info("Validating training and testing data...")
            if not self.check_column_count(train_df, len(columns)):
                debug_message += "Training data column count mismatch. "
                raise MyException("Training data column count mismatch.", sys)

            if not self.check_column_count(test_df, len(columns)):
                debug_message += "Testing data column count mismatch. "
                raise MyException("Testing data column count mismatch.", sys)

            if not self.allowed_columns(train_df, allowed):
                debug_message += "Training data contains disallowed columns. "
                raise MyException("Training data contains disallowed columns.", sys)
            
            if not self.allowed_columns(test_df, allowed):
                debug_message += "Testing data contains disallowed columns. "
                raise MyException("Testing data contains disallowed columns.", sys)
            validation_status = True
            logging.info("Data validation completed successfully.")
        except Exception as e:
            raise MyException(e, sys) from e
        finally:
            os.makedirs(self.data_validation_dir, exist_ok=True)
            report = {
                "validation_status": validation_status,
                "debug_message": debug_message
            }
            with open(self.report_file_path, 'w') as report_file:
                yaml.dump(report, report_file)
            logging.info(f"Data validation report saved at {self.report_file_path}")
            return DataValidationArtifact(
                status=validation_status,
                message=debug_message,
                report_file_path=self.report_file_path
            )