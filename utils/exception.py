# automdl/utils/exceptions.py

from typing import Optional


class AutoDMLError(Exception):
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[str] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.details = details

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        base_msg = f"[AutoDML Error]"

        if self.error_code:
            base_msg += f" [{self.error_code}]"

        base_msg += f" {self.message}"

        if self.details:
            base_msg += f" | Details: {self.details}"

        return base_msg

    def __str__(self):
        return self._format_message()


class DataValidationError(AutoDMLError):
    """Raised when dataset validation fails"""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message, error_code="DATA_VALIDATION_ERROR", details=details
        )


class PreprocessingError(AutoDMLError):
    """Raised during preprocessing failure"""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message, error_code="PREPROCESSING_ERROR", details=details
        )


class ModelTrainingError(AutoDMLError):
    """Raised when model training fails"""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(
            message=message, error_code="MODEL_TRAINING_ERROR", details=details
        )
