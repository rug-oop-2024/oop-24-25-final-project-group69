import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from typing import TypeVar

File = TypeVar("File")

def create(uploaded_file: File) -> Dataset:
    """
    Creates a Dataset instance from the uploaded CSV file.

    Args:
        uploaded_file (UploadedFile): The CSV file uploaded by the user.

    Returns:
        Dataset: A Dataset instance created from the uploaded CSV file.
    """
    df = pd.read_csv(uploaded_file)

    file_name = uploaded_file.name.split('.')[0]
    file_path = uploaded_file.name

    # Create a Dataset instance using the DataFrame
    dataset = Dataset.from_dataframe(
        data=df,
        name=file_name,
        asset_path=file_path
    )

    return dataset


def save(dataset: Dataset) -> None:
    """
    Saves the given Dataset instance using the AutoML system.

    Args:
        dataset (Dataset): The Dataset instance to be saved.

    Returns:
        None
    """
    automl.registry.register(dataset)


automl = AutoMLSystem.get_instance()

st.write("Datasets")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Create and save the dataset only if a file is uploaded
    dataset = create(uploaded_file)
    save(dataset)
