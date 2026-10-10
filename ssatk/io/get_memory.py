import os


def get_memory_usage() -> str:
    """
    Get the current memory usage of the process in GB.

    Returns
    -------
    str
        Formatted memory used by the process in gigabytes.
    """
    try:
        from psutil import Process
    except ImportError:
        return (
            "Memory usage unavailable; install ssatk[monitoring] "
            "to enable it."
        )

    memory_used = Process(os.getpid()).memory_info().rss / 1024 ** 3
    return f"Memory used: {memory_used:.2f} GB"
