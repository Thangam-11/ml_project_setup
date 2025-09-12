import sys

class MLProjectException(Exception):
    """
    Custom exception handler for ML projects.
    Captures error message, file name, and line number.
    """
    def __init__(self, error_message: str, error_detail: Exception):
        super().__init__(error_message)
        try:
            _, _, exc_tb = sys.exc_info()
            self.error_message = error_message
            self.lineno = exc_tb.tb_lineno if exc_tb else 'Unknown'
            self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else 'Unknown'
        except Exception:
            self.lineno = 'Unknown'
            self.file_name = 'Unknown'
        self.original_exception = error_detail

    def __str__(self):
        return (
            f"\n🚨 ML Project Exception:\n"
            f"📄 File: {self.file_name}\n"
            f"🔢 Line: {self.lineno}\n"
            f"💬 Message: {self.error_message}\n"
            f"⚠️ Original Error: {self.original_exception}"
        )
