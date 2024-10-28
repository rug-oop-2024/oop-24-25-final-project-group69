import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

from sklearn.datasets import load_iris

automl = AutoMLSystem.get_instance()

# this is useless imo
datasets = automl.registry.list(type="dataset")

# your code here
st.title("Datasets")

#load the iris dataset
iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)
dataset = Dataset.from_dataframe(
    name="iris",
    asset_path="iris.csv",
    data=df, 
)

#saving the dataset
automl.registry.register(dataset)