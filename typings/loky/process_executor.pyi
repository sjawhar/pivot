"""Type stubs for loky.process_executor internals."""

def _python_exit() -> None:
    """Atexit handler that waits for worker threads to finish."""
    ...
