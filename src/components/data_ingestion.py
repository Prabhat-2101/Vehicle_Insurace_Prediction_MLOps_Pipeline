import os,sys, pymongo
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from src.utils.mongo_helper import connect_to_mongo
from src.utils.logger import logging 
from src.utils.exception_handler import MyException
from datetime import datetime
from src.entity import DataIngestionArtifact

load_dotenv()

class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion class with database and directory configurations.
        """
        self.db_name = os.getenv("DATA_INGESTION_DB_NAME")
        self.collection_name = os.getenv("DATA_INGESTION_COLLECTION_NAME")
        self.artifacts_dir = os.getenv("DATA_ROOT_DIR")
        self.date_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.data_ingestion_dir = os.path.join( self.artifacts_dir,self.date_dir,os.getenv("DATA_INGESTION_DIR_NAME"))
        self.feature_store_dir = os.path.join(self.data_ingestion_dir,os.getenv("DATA_INGESTION_FEATURE_STORE_DIR"))
        self.ingested_dir = os.path.join(self.data_ingestion_dir,os.getenv("DATA_INGESTION_INGESTED_DIR"))
        self.train_test_split_ratio = float(os.getenv("DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO"))
        logging.info("DataIngestion class initialized successfully.")

    def export_data_to_feature_store(self):
        """
        Exports data from MongoDB to a feature store as a CSV file.
        """
        try:
            client = connect_to_mongo()
            db = client[self.db_name]
            collection = db[self.collection_name]
            data = list(collection.find({},{'_id': 0}))
            df = pd.DataFrame(data)

            os.makedirs(self.artifacts_dir, exist_ok=True)
            os.makedirs(self.feature_store_dir, exist_ok=True)

            feature_store_path = os.path.join(self.feature_store_dir, "data.csv")
            df.to_csv(feature_store_path, index=False)

            logging.info(f"Data exported to feature store at {feature_store_path}")
            return feature_store_path
        except Exception as e:
            logging.error(f"Error exporting data to feature store: {e}")
            raise MyException(e,sys)

    def split_data_into_train_test(self, feature_store_path):
        """
        Splits the data from the feature store into training and testing datasets.
        """
        try:
            df = pd.read_csv(feature_store_path)

            train_df, test_df = train_test_split(df, test_size=self.train_test_split_ratio, random_state=42)

            os.makedirs(self.ingested_dir, exist_ok=True)

            train_path = os.path.join(self.ingested_dir, "train.csv")
            test_path = os.path.join(self.ingested_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(f"Data splitted into train and test sets at {train_path} and {test_path}")
            return train_path, test_path
        except Exception as e:
            logging.error(f"Error splitting data into train and test sets: {e}")
            raise MyException(e, sys)

    def run(self):
        try:
            logging.info("Starting data ingestion process...")
            feature_store_path = self.export_data_to_feature_store()
            train_path, test_path = self.split_data_into_train_test(feature_store_path)
            logging.info("Data ingestion process completed successfully.")
            return DataIngestionArtifact(
                date_dir=self.date_dir,
                train_file_path=train_path,
                test_file_path=test_path
            )
        except Exception as e:
            logging.error(f"Error in data ingestion process: {e}")
            raise MyException(e, sys)