import os,sys
from src.utils.exception_handler import MyException
from src.utils.logger import logging
from src.entity import (
    ModelTrainingArtifact,
    ModelEvaluationArtifact
)
from src.utils.s3_operations import S3Operations
from src.utils.common import read_yaml_file
from dotenv import load_dotenv

load_dotenv()

class ModelEvaluation:
    """
    Model Evaluation class to compare the newly trained model with the production model stored in S3.
    """
    def __init__(self):
        self.s3 = S3Operations()
        self.metric_name = os.getenv("PRIMARY_METRIC")
        self.production_metric_key = "models/production/metrics.yaml"

    def run(
        self,
        model_trainer_artifact: ModelTrainingArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation")

            current_metrics = read_yaml_file(
                model_trainer_artifact.metrics_file_path
            )
            current_score = current_metrics[self.metric_name]

            logging.info(f"Current model {self.metric_name}: {current_score}")

            if not self.s3.file_exists(self.production_metric_key):
                logging.info("No production model found. Accepting first model.")

                return ModelEvaluationArtifact(
                    push_model=True,
                    trained_model_path=model_trainer_artifact.model_file_path,
                    best_model_metric=current_score
                )

            prod_metrics = self.s3.read_yaml_file(self.production_metric_key)
            prod_score = prod_metrics[self.metric_name]

            logging.info(f"Production model {self.metric_name}: {prod_score}")

            push_model = current_score > prod_score

            logging.info(f"Model comparison result → Push: {push_model}")

            return ModelEvaluationArtifact(
                push_model=push_model,
                trained_model_path=model_trainer_artifact.model_file_path,
                best_model_metric=max(current_score, prod_score)
            )

        except Exception as e:
            raise MyException(e, sys)
