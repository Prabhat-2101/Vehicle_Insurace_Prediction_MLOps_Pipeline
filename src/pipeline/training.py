import os, sys
from src.utils.exception_handler import MyException
from src.utils.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation

class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Starting training pipeline...")
            data_ingestion = DataIngestion()
            data_ingestion_artifact = data_ingestion.run()
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_validation_artifact = data_validation.run()
            if data_validation_artifact.validation_status == False:
                raise MyException("Data Validation Failed. Check logs for details.", sys)
            data_transform = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact
            )
            data_transform_artifact = data_transform.run()
            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise MyException(e, sys)