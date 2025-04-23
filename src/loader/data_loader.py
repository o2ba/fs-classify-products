import pandas as pd
import os


def load_dataset(filepath, drop_id=True) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.

    Parameters:
    ----------
    filepath : str
        Path to the CSV file.
    drop_id : bool
        Whether to drop the 'id' column (recommended to avoid confusing the model).

    Returns:
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    df = pd.read_csv(filepath)

    if drop_id and 'id' in df.columns:
        df = df.drop(columns=['id'])

    return df