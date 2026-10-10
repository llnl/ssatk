from typing import List, Dict, Optional, Union
from pandas import read_csv as pd_read_csv, DataFrame, errors, concat
from .guess_delimiter import guess_csv_delimiter
import numpy as np
import csv
import os
from .io_utils import exists
from ssatk._paths import ensure_file_parent


def read_csv(file_name: str, sep: Optional[str] = None, dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None,
             col: Union[bool, List[str], None] = False, to_np: bool = False, drop_nan: bool = False,
             skiprows: Optional[List[int]] = None) -> Union[DataFrame, np.ndarray]:
    """
    Read a CSV file with options.

    Args:
        file_name (str): The path to the CSV file.
        sep (Optional[str]): The delimiter used in the CSV file.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): Dictionary specifying data types for columns.
        col (Union[bool, List[str], None]): Specify columns to read.
        to_np (bool): Convert the loaded data to a NumPy array.
        drop_nan (bool): Drop rows with missing values.
        skiprows (List[int]): Rows to skip while reading the CSV file.

    Returns:
        Union[pd.DataFrame, np.ndarray]: A DataFrame or NumPy array with the loaded data.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if skiprows is None:
        skiprows = []

    if col and not isinstance(col, list):
        col = [col]  # Ensure col is always a list

    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter

    if col is False:
        try:
            df = pd_read_csv(file_name, sep=sep, on_bad_lines='skip', skiprows=skiprows, dtype=dtypes)
        except TypeError:
            df = pd_read_csv(file_name, sep=sep, skiprows=skiprows, dtype=object)
    else:
        try:
            if not isinstance(col, list):
                col = [col]
            df = pd_read_csv(file_name, sep=sep, usecols=col, on_bad_lines='skip', skiprows=skiprows, dtype=dtypes)
        except TypeError:
            df = pd_read_csv(file_name, sep=sep, usecols=col, skiprows=skiprows, dtype=object)

    if drop_nan:
        df = df.dropna()

    if to_np:
        return np.squeeze(df.to_numpy())
    else:
        return df


def makedf(df: Union[DataFrame, List, Dict]) -> DataFrame:
    """
    Converts a list or dictionary into a Pandas DataFrame.

    Args:
        df (Union[pd.DataFrame, List, Dict]): A DataFrame, list, or dictionary.

    Returns:
        pd.DataFrame: A DataFrame created from the input.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if isinstance(df, (list, dict)):
        return DataFrame.from_dict(df)
    else:
        return df


def save_csv(file_name: str, df: DataFrame, sep: str = ',', dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None) -> None:
    """
    Save a Pandas DataFrame to a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        df (pd.DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): A dictionary specifying data types for columns.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    df = makedf(df)

    if dtypes:
        df = df.astype(dtypes)

    ensure_file_parent(file_name)
    df.to_csv(file_name, index=False, sep=sep)
    print(f'Saved {file_name} successfully.')
    return


def read_csv_header(file_name: str, sep: Optional[str] = None) -> List[str]:
    """
    Get the header of a CSV file.

    Args:
        file_name (str): The filename of the CSV file.
        sep (Optional[str]): The delimiter used in the CSV file.

    Returns:
        List[str]: A list of the header fields.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    if sep is None:
        sep = guess_csv_delimiter(file_name)  # Guess the delimiter
    with open(file_name, 'r') as infile:
        reader = csv.DictReader(infile, delimiter=sep)
        fieldnames = reader.fieldnames
    return fieldnames


def save_csv_header(filename: str, header: List[str], delimiter: str = ',') -> None:
    """
    Saves a header row to a CSV file with a specified delimiter.

    Args:
        filename (str): The name of the file where the header will be saved.
        header (List[str]): A list of strings representing the column names.
        delimiter (str, optional): The delimiter to use between columns in the CSV file. Default is comma (',').

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    ensure_file_parent(filename)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(header)


def append_csv(file_names: List[str], save_path: str = 'combined_data.csv', sep: str = ',',
               dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None, progress: Optional[callable] = None) -> None:
    """
    Appends multiple CSV files into a single CSV file.

    Args:
        file_names (List[str]): A list of CSV file names.
        save_path (str): The path to the output CSV file. If not specified, the output will be saved to the current working directory.
        sep (Optional[str]): The delimiter used in the CSV files. If None, delimiter will be guessed.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): A dictionary specifying data types for columns.
        progress (Optional[callable]): When provided, print progress for each input file.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    error_files = []
    dataframes = []
    for i, file in enumerate(file_names):
        try:
            df = pd_read_csv(file, sep=guess_csv_delimiter(file))
            dataframes.append(df)
            if progress is not None:
                print(f"Appended {i+1} of {len(file_names)}. File: {file}")
        except (FileNotFoundError, errors.EmptyDataError, errors.ParserError) as e:
            error_files.append(file)
            print(f"Error processing file {file}: {e}")

    combined_df = concat(dataframes, ignore_index=True)
    if dtypes:
        combined_df = combined_df.astype(dtypes)

    if save_path:
        ensure_file_parent(save_path)
        combined_df.to_csv(save_path, sep=sep, index=False)
    else:
        ensure_file_parent('combined_data.csv')
        combined_df.to_csv('combined_data.csv', sep=sep, index=False)

    print(f'The final dataframe has {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.')
    if error_files:
        print(f'The following files ERRORED and were not included: {error_files}')
    return


def append_csv_on_disk(csv_files: List[str], output_file: str) -> None:
    """
    Appends multiple CSV files directly to a single output file on disk.

    Args:
        csv_files (List[str]): A list of CSV files to append.
        output_file (str): The path to the output CSV file.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    delimiter = guess_csv_delimiter(csv_files[0])
    # Open the output file for writing
    ensure_file_parent(output_file)
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter)

        # Write the header row from the first CSV file
        with open(csv_files[0], 'r', newline='') as first_file:
            reader = csv.reader(first_file, delimiter=delimiter)
            header = next(reader)
            writer.writerow(header)

            # Write the data rows from the first CSV file
            for row in reader:
                writer.writerow(row)

        # Write the data rows from the remaining CSV files
        for file in csv_files[1:]:
            with open(file, 'r', newline='') as infile:
                reader = csv.reader(infile, delimiter=delimiter)
                next(reader)  # Skip the header row
                for row in reader:
                    writer.writerow(row)
    print(f'Completed appending of: {output_file}.')


def append_dict_to_csv(file_name: str, data_dict: Dict[str, List[Union[str, float, int]]], delimiter: str = ',') -> None:
    """
    Appends data from a dictionary to a CSV file.

    Args:
        file_name (str): The path to the CSV file.
        data_dict (Dict[str, List[Union[str, float, int]]]): A dictionary where keys are column names and values are lists of column data.
        delimiter (str): The delimiter used in the CSV file.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    # Check if the input is a numpy array or DataFrame, and convert to dictionary if necessary
    if isinstance(data_dict, np.ndarray):
        # Convert ndarray to dictionary (assuming each column is a field)
        data_dict = {f'col{i}': data_dict[:, i].tolist() for i in range(data_dict.shape[1])}
    elif isinstance(data_dict, DataFrame):
        # Convert DataFrame to dictionary (using columns as keys)
        data_dict = data_dict.to_dict(orient='list')

    # Extract keys and values from the dictionary
    keys = list(data_dict.keys())
    values = list(data_dict.values())

    # Determine the length of the arrays
    array_length = len(values[0])

    # Determine if the file exists
    file_exists = os.path.exists(file_name)

    # Open the CSV file in append mode
    ensure_file_parent(file_name)
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)

        # Write header if the file doesn't exist
        if not file_exists:
            writer.writerow(keys)

        # Write each element from arrays as a new row
        for i in range(array_length):
            row = [values[j][i] for j in range(len(keys))]
            writer.writerow(row)


def save_csv_array_to_line(filename: str, array: List[Union[str, float, int]], delimiter: str = ',') -> None:
    """
    Appends a single row of data to a CSV file with a specified delimiter.

    Args:
        filename (str): The name of the file to which the row will be appended.
        array (List[Union[str, float, int]]): A list of values representing a single row of data to be appended to the CSV file.
        delimiter (str, optional): The delimiter to use between columns in the CSV file. Default is comma (',').

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    ensure_file_parent(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)
        writer.writerow(array)


def save_csv_line(file_name: str, df: DataFrame, sep: str = ',', dtypes: Optional[Dict[str, Union[str, np.dtype]]] = None) -> None:
    """
    Save a Pandas DataFrame to a CSV file, appending the DataFrame to the file if it exists.

    Args:
        file_name (str): The path to the CSV file.
        df (pd.DataFrame): The Pandas DataFrame to save.
        sep (str): The delimiter used in the CSV file.
        dtypes (Optional[Dict[str, Union[str, np.dtype]]]): A dictionary specifying data types for columns.

    Returns:
        None

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    df = makedf(df)
    if dtypes:
        df = df.astype(dtypes)
    if exists(file_name):
        df.to_csv(file_name, mode='a', index=False, header=False, sep=sep)
    else:
        save_csv(file_name, df, sep=sep)
    return


_column_data = None


def exists_in_csv(csv_file: str, column: str, number: Union[int, float, str], sep: str = ',') -> bool:
    """
    Checks if a number exists in a specific column of a CSV file.

    Args:
        csv_file (str): The path to the CSV file.
        column (str): The column name to search.
        number (Union[int, float, str]): The value to search for.
        sep (str, optional): The delimiter used in the CSV file. Default is comma (',').

    Returns:
        bool: True if the number exists in the column, False otherwise.

    Author: Travis Yeager (yeager7@llnl.gov)
    """
    try:
        global _column_data
        if _column_data is None:
            _column_data = read_csv(csv_file, sep=sep, col=column, to_np=True)
        return np.isin(number, _column_data)
    except IOError:
        return False
