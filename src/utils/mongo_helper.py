from pymongo.mongo_client import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv
import os, sys
import pandas as pd
from .logger import logging
from .exception_handler import MyException

load_dotenv()

def connect_to_mongo():
    """
    Establishes a connection to the MongoDB database.
    """
    try:
        string  = os.getenv("CONNECTION_STRING")
        client = MongoClient(string)
        client.admin.command("ping")
        logging.info("Successfully connected to MongoDB.")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logging.error(f"Error connecting to MongoDB: {e}")
        raise MyException(e, sys)

def push_data_to_mongo(data_path: str):
    """
    Pushes data from a CSV file to the specified MongoDB collection.
    """
    try:
        client = connect_to_mongo()

        string = os.getenv("CONNECTION_STRING")
        db_name = os.getenv("DB_NAME")
        collection_name = os.getenv("COLLECTION_NAME")

        db = client[db_name]
        collection = db[collection_name]

        df = pd.read_csv(data_path)
        data = df.to_dict(orient="records")

        res = collection.insert_many(data)
        logging.info(f"Inserted {len(res.inserted_ids)} records into the collection.")

    except Exception as e:
        logging.error(f"Error occurred while inserting data: {e}")
        raise MyException(e, sys)

if __name__ == "__main__":
    push_data_to_mongo("data.csv")