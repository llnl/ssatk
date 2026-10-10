import csv
from typing import Optional, Sequence


def guess_csv_delimiter(
    csv_file_path: str,
    sample_size: int = 32768,
    delimiters: Optional[Sequence[str]] = None,
) -> str:
    """
    Guesses the delimiter used in a CSV file.

    Args:
        csv_file_path (str): The path to the CSV file.
        sample_size (int): The number of bytes to read from the file to guess the delimiter.
        delimiters (Sequence[str] | None): Candidate delimiters to test.

    Returns:
        str: The detected delimiter or an error message if unable to detect.
    """
    if delimiters is None:
        delimiters = (',', ';', '\t', '|', ' ')

    with open(csv_file_path, 'r') as csvfile:
        sample = csvfile.read(sample_size)  # Read a larger sample size
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=delimiters)
            return dialect.delimiter
        except csv.Error:
            return "Could not determine delimiter"
