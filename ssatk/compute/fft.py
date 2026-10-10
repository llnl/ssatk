import numpy as np
from ..utils import divby0


def FFT(data: np.ndarray, time_between_samples: float = 1) -> np.ndarray:
    """
    Perform a Fast Fourier Transform on the input data.

    Parameters:
    data (np.ndarray): The input time-series data.
    time_between_samples (float): Time interval between data samples.

    Returns:
    tuple: The frequency array and the FFT result.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    N = len(data)
    k = N // 2
    # Bin k of an N-point DFT is at k / (N dt); linspace(0, 1/(2 dt), N//2)
    # stretched the axis by N / (N - 2).
    f = np.arange(k) / (N * time_between_samples)
    Y = np.abs(np.fft.fft(data))[:k]
    return f, Y


def FFTP(data: np.ndarray, time_between_samples: float = 1) -> np.ndarray:
    """
    Perform a Fast Fourier Transform and calculate the period.

    Parameters:
    data (np.ndarray): The input time-series data.
    time_between_samples (float): Time interval between data samples.

    Returns:
    tuple: The period array and the FFT result.

    Author: Travis Yeager (yaeger7@llnl.gov)
    """
    N = len(data)
    k = N // 2
    f = np.arange(k) / (N * time_between_samples)
    Tp = [divby0(1, float(item), len(data) * time_between_samples) for item in f]
    Y = np.abs(np.fft.fft(data))[:k]
    return Tp, Y
