import os,sys
from src.utils.exception_handler import MyException
from src.utils.logger import logging
from src.entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelPusherArtifact,
    ModelTrainingArtifact
)
from src.utils.s3_operations import S3Operations
from dotenv import load_dotenv

load_dotenv()

class ModelPusher:
    def __init__(self):
        self.s3 = S3Operations()
        self.model_name = os.getenv("MODEL_NAME")

    def _get_next_version(self) -> int:
        """
        Determine next model version (v1, v2, ...)
        """
        try:
            response = self.s3.s3.list_objects_v2(
                Bucket=self.s3.bucket,
                Prefix=f"models/registry/{self.model_name}/v"
            )

            if "Contents" not in response:
                return 1

            versions = []
            for obj in response["Contents"]:
                parts = obj["Key"].split("/")
                if len(parts) >= 4 and parts[3].startswith("v"):
                    versions.append(int(parts[3][1:]))

            return max(versions) + 1 if versions else 1

        except Exception:
            return 1

    def run(
        self,
        model_eval_artifact: ModelEvaluationArtifact,
        data_transform_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainingArtifact
    ) -> ModelPusherArtifact:
        try:
            if not model_eval_artifact.push_model:
                logging.info("Model rejected during evaluation. Skipping model push.")
                return None

            logging.info("Starting Model Pusher")

            version = self._get_next_version()
            version_path = f"models/registry/{self.model_name}/v{version}"
            production_path = f"models/registry/{self.model_name}/production"

            # Upload model
            self.s3.upload_file(
                model_eval_artifact.trained_model_path,
                f"{version_path}/model.pkl"
            )
            self.s3.upload_file(
                model_eval_artifact.trained_model_path,
                f"{production_path}/model.pkl"
            )

            # Upload preprocessor
            self.s3.upload_file(
                data_transform_artifact.preprocessed_object_file_path,
                f"{version_path}/preprocessor.pkl"
            )
            self.s3.upload_file(
                data_transform_artifact.preprocessed_object_file_path,
                f"{production_path}/preprocessor.pkl"
            )

            # Upload metrics
            self.s3.upload_file(
                model_trainer_artifact.metrics_file_path,
                f"{version_path}/metrics.yaml"
            )
            self.s3.upload_file(
                model_trainer_artifact.metrics_file_path,
                f"{production_path}/metrics.yaml"
            )

            logging.info(
                f"Model pushed successfully. Version: v{version} promoted to production."
            )

            return ModelPusherArtifact(
                model_eval_artifact=model_eval_artifact,
                data_transform_artifact=data_transform_artifact,
                model_trainer_artifact=model_trainer_artifact
            )

        except Exception as e:
            raise MyException(e, sys)
