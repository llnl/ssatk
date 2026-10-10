"""Helpers for distributing work across CPUs or nodes.

These utilities provide simple index-based partitioning that works with or
without MPI.
"""


def get_unique_id(rank, run_number, cpus_per_node):
    """
    Calculates a unique ID based on the rank, run number, and CPUs per node.

    Parameters:
    rank (int): The rank of the CPU within the node.
    run_number (int): The current run number.
    cpus_per_node (int): The number of CPUs per node.

    Returns:
    int: The unique ID for the given rank, run number, and CPUs per node.
    """
    unique_id = rank + cpus_per_node * run_number
    return unique_id


def distribute_array_no_mpi(unique_id, total_jobs, array_size):
    """Compute a contiguous slice of work for a given worker.

    Parameters
    ----------
    unique_id : int
        Unique worker identifier (e.g., from ``get_unique_id``).
    total_jobs : int
        Total number of workers sharing the work.
    array_size : int
        Length of the 1D array being partitioned.

    Returns
    -------
    tuple[int | None, int | None]
        ``(start_idx, end_idx)`` for the slice assigned to the worker,
        or ``(None, None)`` if no work is assigned.
    """
    chunk_size = array_size // total_jobs
    remainder = array_size % total_jobs

    if unique_id < remainder:
        start_idx = unique_id * (chunk_size + 1)
        end_idx = start_idx + chunk_size + 1
    else:
        start_idx = unique_id * chunk_size + remainder
        end_idx = start_idx + chunk_size

    if start_idx >= array_size:
        return None, None

    return int(start_idx), int(end_idx)
