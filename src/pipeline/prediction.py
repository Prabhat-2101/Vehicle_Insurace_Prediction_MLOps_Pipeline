import os, joblib
import boto3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

class PredictionPipeline:
    """
    PredictionPipeline class for making predictions using a pre-trained model stored in S3.
    """
    def __init__(self):
        self.bucket = os.getenv("AWS_S3_BUCKET_NAME")
        self.model_name = os.getenv("MODEL_NAME")
        self.s3 = boto3.client("s3")

        self.base_key = f"models/registry/{self.model_name}/production"

        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()

    def _load_model(self):
        if not self.bucket or not self.model_name:
            raise ValueError("AWS_S3_BUCKET_NAME or MODEL_NAME is not set")

        key = f"{self.base_key}/model.pkl"

        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            body = obj["Body"].read()

            if body is None:
                raise ValueError("S3 returned empty model file")

            return joblib.load(BytesIO(body))

        except Exception as e:
            raise RuntimeError(f"Failed to load model from S3: {str(e)}")


    def _load_preprocessor(self):
        obj = self.s3.get_object(
            Bucket=self.bucket,
            Key=f"{self.base_key}/preprocessor.pkl"
        )
        return joblib.load(BytesIO(obj["Body"].read()))

    def predict(self, input_data: dict) -> dict:
        """
        input_data → raw form input dict
        returns → prediction + probability
        """

        df = pd.DataFrame([input_data])

        transformed_data = self.preprocessor.transform(df)

        prediction = int(self.model.predict(transformed_data)[0])
        probability = float(self.model.predict_proba(transformed_data)[0][1])

        return {
            "prediction": prediction,
            "probability": probability
        }
