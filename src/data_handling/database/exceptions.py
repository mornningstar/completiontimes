class DatabaseError(Exception):
    """Base class for database exceptions."""
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception