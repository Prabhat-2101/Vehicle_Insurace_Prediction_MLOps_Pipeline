import boto3
import os
from botocore.exceptions import ClientError
from .exception_handler import MyException
from dotenv import load_dotenv

load_dotenv()

class S3Operations:
    def __init__(self):
        self.s3 = boto3.client("s3")
        self.bucket = os.getenv("AWS_S3_BUCKET_NAME")

        if not self.bucket:
            raise ValueError("AWS_S3_BUCKET_NAME environment variable not set")

    def test_connection(self) -> bool:
        """Check if bucket is accessible"""
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            return True
        except ClientError as e:
            logging.error(f"S3 connection error: {e}")
            return False


    def file_exists(self, s3_key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False

    def create_directory(self, directory_name: str):
        """Create a folder (prefix) in S3"""
        if not directory_name.endswith("/"):
            directory_name += "/"

        self.s3.put_object(Bucket=self.bucket, Key=directory_name)

    def upload_file(self, local_path: str, s3_key: str):
        self.s3.upload_file(local_path, self.bucket, s3_key)

    def download_file(self, s3_key: str, local_path: str):
        self.s3.download_file(self.bucket, s3_key, local_path)
