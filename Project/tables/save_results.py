"""
Results persistence utilities for VQE benchmark experiments.

Provides :func:`save_results` for writing pandas DataFrames to CSV.
Directories are created automatically. Append mode is supported so
that results from successive runs accumulate in a single file without
overwriting earlier entries.
"""

from __future__ import annotations

import os

import pandas as pd


def save_results(
    filepath: str,
    dataframe: pd.DataFrame,
    mode: str = "a",
    include_header: bool | None = None,
) -> str:
    """Save a benchmark summary DataFrame to a CSV file.

    Creates all parent directories automatically. In append mode (``'a'``),
    the header row is written only if the target file does not yet exist or
    is empty, preventing duplicate header lines in multi-run accumulation.

    Args:
        filepath: Absolute or relative path to the output ``.csv`` file.
            The extension is not enforced but ``.csv`` is recommended.
        dataframe: Pandas DataFrame containing one or more benchmark rows
            (output of :func:`analysis.create_summary_table`).
        mode: File write mode.

            - ``'a'`` (default) — append; header written only for new/empty
              files.
            - ``'w'`` — overwrite; header always written.

        include_header: Override the automatic header logic. If ``True``,
            always write the header; if ``False``, never write it; if
            ``None`` (default), apply the ``mode``-based rules above.

    Returns:
        The resolved absolute path of the CSV file that was written.

    Raises:
        ValueError: If ``mode`` is not ``'w'`` or ``'a'``.
        OSError: If the file cannot be written (e.g. permission denied).

    Example:
        >>> from pathlib import Path
        >>> path = save_results("results/h2o_uccsd.csv", df)
        >>> print(path)
        /absolute/path/to/results/h2o_uccsd.csv

        >>> # Accumulate results across runs in one file
        >>> for config in configs:
        ...     run_result_df = run_vqe(config)
        ...     save_results("results/sweep.csv", run_result_df, mode="a")
    """
    if mode not in ("w", "a"):
        raise ValueError(
            f"mode must be 'w' (overwrite) or 'a' (append), got '{mode}'."
        )

    resolved = os.path.abspath(filepath)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if include_header is None:
        if mode == "w":
            include_header = True
        else:
            # Append: write header only when the file is new or empty
            include_header = (
                not os.path.isfile(resolved)
                or os.path.getsize(resolved) == 0
            )

    dataframe.to_csv(resolved, mode=mode, index=False, header=include_header)

    return resolved


def load_results(filepath: str) -> pd.DataFrame:
    """Load a previously saved benchmark CSV into a DataFrame.

    Convenience wrapper around :func:`pandas.read_csv` with sensible
    defaults for the benchmark file format.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Loaded :class:`pandas.DataFrame`.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> df = load_results("results/h2o_uccsd.csv")
        >>> df[["Ansatz", "Error (mHa)", "Runtime (s)"]]
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Results file not found: {os.path.abspath(filepath)}"
        )
    return pd.read_csv(filepath)
