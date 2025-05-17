import pandas as pd
import csv

'''
This file is meant to load and format the specific data in the Datasets folder
with a unique callable function from the main file
'''


# Define the full column list for wdbc.data
wdbc_columns = [
    'id', 'diagnosis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

# Define column names for adult dataset
adult_columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]


def data_reader(filepath):

    #print(f"Loading data from {filepath}...")

    # Detect delimiter
    with open(filepath, 'r', encoding='utf-8') as f:
        sample = f.read(2048)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        delimiter = dialect.delimiter
    
    # Use custom columns if file is wdbc.data
    if filepath == "Datasets/breast+cancer+wisconsin+diagnostic/wdbc.data":
        df = pd.read_csv(filepath, delimiter=",", header=None, names=wdbc_columns)
    elif filepath == "Datasets/adult/adult.data":
        df_test = pd.read_csv("Datasets/adult/adult.test", names=adult_columns, skiprows=1,skipinitialspace=True)
        df_test['income'] = df_test['income'].str.replace('.', '', regex=False)
        df_train = pd.read_csv(filepath, names=adult_columns, skipinitialspace=True)
        df = pd.concat([df_train, df_test], ignore_index=True)
    else:
        df = pd.read_csv(filepath, delimiter=delimiter)
        # Insert index column if not present
        if not df.columns[0].lower().__str__().startswith('index'):
            df.insert(0, 'index', range(1, len(df) + 1))

    return df

def data_cleaning(df, data=None):

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    # Remove rows with all zero values
    df = df.loc[(df != 0).any(axis=1)]

    # Remove rows with all NaN values
    df = df.dropna(how='all')

    if data is not None:
        
        df.drop(columns=["SMILES"])

    return df
