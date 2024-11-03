from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io


class Dataset(Artifact):
    """
    Represents a dataset artifact, extending the Artifact class with specific methods
    for handling datasets stored as CSVs.

    Inherits from:
        Artifact: The base class for artifacts, with additional dataset-specific functionality.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the Dataset with a type of "dataset".

        Args:
            *args: Positional arguments passed to the Artifact initializer.
            **kwargs: Keyword arguments passed to the Artifact initializer.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame,
                       name: str,
                       asset_path: str,
                       version: str = "1.0.0") -> "Dataset":
        """
        Creates a Dataset instance from a Pandas DataFrame, converting it to CSV format.

        Args:
            data (pd.DataFrame): The data to store in the dataset.
            name (str): The name of the dataset.
            asset_path (str): The storage path for the dataset.
            version (str, optional): The version of the dataset. Defaults to "1.0.0".

        Returns:
            Dataset: A Dataset instance with the DataFrame data stored as CSV.
        """
        return Dataset(
            name = name,
            asset_path = asset_path,
            data = data.to_csv(index=False).encode(),
            version = version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset data as a Pandas DataFrame.

        Returns:
            pd.DataFrame: The data stored in the dataset, loaded from CSV format.
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> None:
        """
        Saves a Pandas DataFrame to the dataset by converting it to CSV format.

        Args:
            data (pd.DataFrame): The DataFrame data to save.
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
