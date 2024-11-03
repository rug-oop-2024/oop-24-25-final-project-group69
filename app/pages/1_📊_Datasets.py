import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

# this is useless imo
datasets = automl.registry.list(type="dataset")

st.write("Datasets")
# your code here

# load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    dataset = Dataset.from_dataframe(
        data=df,
        name="iris",
        asset_path="iris.csv"
    )

    # saving the dataset
    automl.registry.register(dataset)
