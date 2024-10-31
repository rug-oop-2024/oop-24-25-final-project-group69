import streamlit as st
import pandas as pd

from io import BytesIO
from pathlib import Path

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types

st.set_page_config(page_title="Modelling", page_icon="📈")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


# your code here
datasets_by_name = {dataset.name: dataset for dataset in datasets}


if datasets_by_name is not None:
    # Use Streamlit's selectbox to allow the user to select a dataset
    selected_dataset_name = st.selectbox("Select a dataset", list(datasets_by_name))

    selected_dataset = datasets_by_name[selected_dataset_name]

    csv_data_from_storage = automl.registry._storage.load(selected_dataset.asset_path)

    df = pd.read_csv(BytesIO(csv_data_from_storage))

    recreated_dataset = Dataset.from_dataframe(
        data=df,
        name=selected_dataset.name,
        asset_path=selected_dataset.asset_path
    )

    # delete this
    st.write(f"{detect_feature_types(recreated_dataset)}")
