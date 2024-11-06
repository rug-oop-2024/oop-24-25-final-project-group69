import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


def create(uploaded_file) -> Dataset:
    df = pd.read_csv(uploaded_file)

    dataset = Dataset.from_dataframe(
        data=df,
        name="iris",
        asset_path="iris.csv"
    )
    
    return dataset

def save(dataset: Dataset) -> None:
    automl.registry.register(dataset)


automl = AutoMLSystem.get_instance()

# this is useless imo
datasets = automl.registry.list(type="dataset")

st.write("Datasets")

# load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    dataset = create(uploaded_file)
    save(dataset)
