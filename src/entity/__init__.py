class DataIngestionArtifact:
    def __init__(self, date_dir: str, train_file_path: str, test_file_path: str):
        self.date_dir = date_dir
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path

class DataValidationArtifact:
    def __init__(self, status:bool, message: str, report_file_path: str):
        self.validation_status = status
        self.debug_message = message
        self.report_file_path = report_file_path

class DataTransformationArtifact:
    def __init__(self, transformed_train_file_path: str, transformed_test_file_path: str, preprocessed_object_file_path: str):
        self.transformed_train_file_path = transformed_train_file_path
        self.transformed_test_file_path = transformed_test_file_path
        self.preprocessed_object_file_path = preprocessed_object_file_path

class ModelTrainingArtifact:
    def __init__(self, model_file_path: str, metrics_file_path: str):
        self.model_file_path = model_file_path
        self.metrics_file_path = metrics_file_path