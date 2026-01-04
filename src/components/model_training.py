from src.entity import DataTransformationArtifact, ModelTrainingArtifact, DataIngestionArtifact
from src.utils.common import write_yaml_file, load_numpy_array_data
from src.utils.exception_handler import MyException
from src.utils.logger import logging
import sys, os, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class ModelTraining:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, data_ingestion_artifact:DataIngestionArtifact):
        """ModelTraining class initialized successfully."""
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_dir = os.getenv('MODEL_TRAINER_DIR_NAME')
            self.model_trainer_model_dir = os.getenv('MODEL_TRAINER_MODEL_DIR')
            self.model_trainer_model_name = os.getenv('MODEL_TRAINER_MODEL_FILE_NAME')
        except Exception as e:
            raise MyException(e, sys)

    def create_model(self):
        """
        Create and return a RandomForestClassifier model with hyperparameters from environment variables.
        """
        model = RandomForestClassifier(
            n_estimators = int(os.getenv('MODEL_TRAINER_N_ESTIMATORS')),
            min_samples_split = int(os.getenv('MODEL_TRAINER_MIN_SAMPLES_SPLIT')),
            min_samples_leaf = int(os.getenv('MODEL_TRAINER_MIN_SAMPLES_LEAF')),
            max_depth = int(os.getenv('MODEL_TRAINER_MAX_DEPTH')),
            criterion = os.getenv('MODEL_TRAINER_CRITERION'),
            random_state = int(os.getenv('MODEL_TRAINER_RANDOM_STATE'))
        )
        return model

    def run(self):
        """
        Trains the model using transformed training data and evaluates it on transformed test data.
        """
        try:
            logging.info("Model training started.")
            transformed_train_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_path = self.data_transformation_artifact.transformed_test_file_path
            logging.info(f"Loading transformed training & testing data from: {transformed_train_path} {transformed_test_path}")
            
            train_data = load_numpy_array_data(transformed_train_path)
            test_data = load_numpy_array_data(transformed_test_path)
            
            X_train, y_train = train_data[:,:-1], train_data[:,-1]
            X_test, y_test = test_data[:,:-1], test_data[:,-1]

            logging.info(f"Training model using training data at: {transformed_train_path}")
            model = self.create_model()
            model.fit(X_train, y_train)
            logging.info("Model training completed.")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            logging.info(f"Model evaluation metrics - Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

            model_dir = os.path.join(os.getenv('DATA_ROOT_DIR'), self.data_ingestion_artifact.date_dir, self.model_trainer_dir, self.model_trainer_model_dir)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, self.model_trainer_model_name)
            logging.info(f"Saving trained model at: {model_path}")
            with open(model_path, 'wb') as model_file:
                joblib.dump(model, model_file)
            
            metrics = {
                "Accuracy": accuracy,
                "F1_Score": f1,
                "Precision": precision,
                "Recall": recall
            }
            metrics_file_path = os.path.join(os.getenv('DATA_ROOT_DIR'), self.data_ingestion_artifact.date_dir,self.model_trainer_dir, os.getenv("MODEL_TRAINER_MODEL_PERFORMANCE_FILE_NAME"))
            write_yaml_file(file_path=metrics_file_path, content=metrics)
            logging.info("Model saved successfully.")
            return ModelTrainingArtifact(model_file_path=model_path,metrics_file_path=metrics_file_path)
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise MyException(e, sys)