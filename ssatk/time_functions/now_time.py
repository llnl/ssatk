from datetime import datetime


def now():
    """
    Returns the current time in the format 'YYYY-MM-DD HH:MM'.

    Returns
    -------
    str
        The current date and time as a formatted string.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M')
