"""Type stubs for wcwidth - terminal width calculation library."""

def wcswidth(pwcs: str, n: int | None = None) -> int:
    """Return display width of string, or -1 if contains non-printable characters."""
    ...

def wcwidth(wc: str) -> int:
    """Return display width of a single character."""
    ...
