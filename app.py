import streamlit as st
from DataFilesNormalization import data_reader
from DataExploreTransform import data_exploration, feature_relationships, feature_engineering
import inspect
import pennylane as qml
from VariationalQuantumAlgorithm import variational_circuit

st.title("Quantum & AI Edge Computing")
st.write("This is a demo of a Variational Quantum Algorithm application.")

# Dataset descritpion
dataset_title = "Drug Induced Autoimmunity Prediction"
dataset_description = (
    "Describe the dataset."
)

# Show dataset title and description in a text area
st.text_area("Dataset Information", f"Title: {dataset_title}\n\nDescription: {dataset_description}", height=150)

# Open a collapsible section to explore data -
# Needs to be changed for the specific dataset
with st.expander("Show and Explore Dataset"):

    df = data_reader("Datasets/adult/adult.data")
    st.write("Preview of the dataset:")
    st.dataframe(df.head())

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
        feature_engineering(df)
  
st.header("Variational Quantum Classifier")

n_qubits = st.number_input("Select the number of qubits", min_value=1, max_value=10, value=3)
n_layers = st.number_input("Select the number of layers", min_value=1, max_value=10, value=2)

'''
# Define the quantum device, variational ansatz
'''

# Show circuit code and allow user to edit it
default_ansatz_code = inspect.getsource(variational_circuit)
st.code(default_ansatz_code, language='python')
ansatz_code = st.button(
    "Edit the variational ansatz code")

if ansatz_code:
    st.text_area(value = default_ansatz_code, height=300)
    exec(ansatz_code)

'''
Insert a button to run the training + training code
'''

'''
Insert a button to test + testing code
'''

'''
Insert a button for performance metrics and evaluation
'''

'''
Results visualization'''