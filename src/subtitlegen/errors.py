class EmptySubtitleError(RuntimeError):
    """Raised when a pipeline stage would create an empty subtitle output."""


class BackendUnavailableError(RuntimeError):
    """Raised when an optional backend is not installed or supported."""


class BackendOutOfMemoryError(RuntimeError):
    """Raised with actionable guidance after a model allocation failure."""
