import numpy as np


def calculate_errors(data: np.ndarray, CI: float = 0.05) -> np.ndarray:
    """
    Calculate the confidence interval errors for a dataset.

    Parameters:
    data (np.ndarray): The input data array.
    CI (float): The confidence interval, default is 0.05.

    Returns:
    tuple: A tuple containing the error bounds and the median of the data.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    data_median = []
    # Drop NaNs before sorting: np.sort puts them last, so they would shift the
    # upper index while nanmedian ignores them.
    data = np.asarray(data, dtype=float)
    data = np.sort(data[~np.isnan(data)])
    median_ = np.median(data)
    data_median.append(median_)
    err = [median_ - data[int(CI * len(data))], data[int((1 - CI) * len(data))] - median_]
    return err, data_median
