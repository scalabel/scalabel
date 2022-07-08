class Error(Exception):
    """Base class for other exceptions."""

    pass


class GPUError(Error):
    """Raised when there are issues with GPUs."""

    pass
