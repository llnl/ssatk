import os
import logging


def build_logging(id, log_dir):
    """
    Set up a logging system that logs messages to a specified directory with a unique file for each ID.

    Parameters
    ----------
    id : int or str
        A unique identifier to associate with the log file and log messages.
    log_dir : str
        The directory where the log file will be created. All necessary subdirectories will be created if they do not exist.

    Returns
    -------
    None

    Notes
    -----
    - The function ensures the log directory exists and removes any existing log file with the same name to start fresh.
    - Logs are saved in a file named `id_<id>.log` within the specified directory.
    - Each log message includes the ID, a timestamp, and the log message.
    - A custom log formatter is used to inject the ID into log messages.
    - The function prints a message to the console indicating where logging has been initialized.

    Example
    -------
    ```python
    build_logging(id=123, log_dir="/path/to/logs")
    logging.info("This is a test log message.")
    ```
    This would create a log file at `/path/to/logs/id_123.log` and log the message with the given ID.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"id_{id}.log"
    log_filepath = f"{log_dir}/{log_filename}"
    if os.path.exists(log_filepath):
        os.remove(log_filepath)

    # Create a custom log formatter to include the id
    class RankFormatter(logging.Formatter):
        def format(self, record):
            record.id = id  # Add id to the log record
            return super().format(record)

    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format="%(asctime)s [ID: %(id)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Set the custom formatter to include the id
    for handler in logging.getLogger().handlers:
        handler.setFormatter(RankFormatter(handler.formatter._fmt, handler.formatter.datefmt))

    logging.info(f"ID: {id} Logging initialized in {log_dir}.")
    print(f"ID: {id} Logging initialized in {log_dir}.")
    return


def log_print(message, show_prints=True):
    logging.info(message)
    if show_prints:
        print(message)
    return
