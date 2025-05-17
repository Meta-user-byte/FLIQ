import streamlit as st
from DataFilesNormalization import data_reader,data_cleaning
import inspect
import pandas as pd
import pennylane as qml
import numpy as np

st.title("Quantum & AI Edge Computing")
st.write("This is a demo of a Variational Quantum Algorithm application.")

# ---Dataset descritpion---
dataset_title = "Drug Induced Autoimmunity Prediction"
dataset_description = (
    "Describe the dataset."
)

# ---Show dataset title and description in a text area---
st.text_area("Dataset Information", f"Title: {dataset_title}\n\nDescription: {dataset_description}", height=150)

# Open a collapsible section to explore data -
# Needs to be changed for the specific dataset
with st.expander("Show and Explore Dataset (Training Data)"):

    train_df = pd.read_csv("Datasets/drug+induced+autoimmunity+prediction/DIA_trainingset_RDKit_descriptors.csv")
    test_df = pd.read_csv("Datasets/drug+induced+autoimmunity+prediction/DIA_testset_RDKit_descriptors.csv")
    data_cleaning(train_df, data="DIA_trainingset_RDKit_descriptors.csv")
    data_cleaning(test_df, data="DIA_testset_RDKit_descriptors.csv")
    st.write("Preview of the dataset:")
    st.dataframe(train_df.head())

    '''
    if st.button("Data exploration"):
        st.write("Exploring the dataset...")
        data_exploration(df, "continuos")
        data_exploration(df, "categorical")
    if st.button("Feature relationships"):
        st.write("Exploring feature relationships...")
        feature_relationships(df, "continuos")
        feature_relationships(df, "categorical")
    if st.button("Feature engineering"):
        st.write("Performing feature engineering...")
        feature_engineering(df)'''
  
st.header("Variational Quantum Classifier")

# ---Selecting columns with higher correlation and setting constants---
X, X_test = train_df.columns[113:], test_df.columns[113:]
X, X_test = train_df[X], test_df[X_test]

n_train, n_test = X.shape[0], X_test.shape[0]

n_qubits = st.number_input("Select the number of qubits", min_value=1, max_value=10, value=6) # "window_size" previously
n_layers = st.number_input("Select the number of layers", min_value=1, max_value=10, value=3) 

dev = qml.device("default.qubit", wires=n_qubits)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers,n_qubits))

'''
# Define the quantum device, variational ansatz
You can copy and paste your code here or define it in a separate file and import it.
'''


# Show circuit code and allow user to edit it
default_ansatz_code = inspect.getsource() # Insert the ansatz code here
st.code(default_ansatz_code, language='python')
ansatz_code = st.button(
    "Edit the variational ansatz code")

if ansatz_code:
    st.text_area(value = default_ansatz_code, height=300)
    exec(ansatz_code)

'''
Insert a button to run the training code
'''

'''
Insert a button to test + testing code
'''

'''
Insert a button for performance metrics and evaluation
'''

'''
Results visualization
'''