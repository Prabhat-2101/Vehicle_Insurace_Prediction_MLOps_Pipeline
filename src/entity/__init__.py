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