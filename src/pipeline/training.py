import os, sys
from src.utils.exception_handler import MyException
from src.utils.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

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
            model_trainer = ModelTraining(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_artifact=data_transform_artifact
            )
            model_trainer_artifact = model_trainer.run()
            model_evaluation = ModelEvaluation()
            model_eval_artifact = model_evaluation.run(
                model_trainer_artifact
            )
            if model_eval_artifact.push_model:
                model_pusher = ModelPusher()
                model_pusher.run(model_eval_artifact,data_transform_artifact, model_trainer_artifact)
                logging.info("New model accepted and pushed to production.")

            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error in training pipeline: {e}")
            raise MyException(e, sys)