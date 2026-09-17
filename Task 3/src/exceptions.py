class OrderValidationError(Exception):
    def __init__(self, message, failed_expectations=None):
        super().__init__(message)
        self.failed_expectations = failed_expectations or []
