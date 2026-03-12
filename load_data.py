import pandas as pd


def load_dataset(file_path):
    """
    Load the CSV dataset into a pandas DataFrame.

    Parameters:
        file_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(file_path)
    return df


def inspect_dataset(df):
    """
    Print basic information about the dataset.

    Parameters:
        df (pd.DataFrame): Input dataframe
    """
    print("\n========== DATASET INSPECTION ==========")
    print("Shape of dataset:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    if "medical_specialty" in df.columns:
        print("\nNumber of unique medical specialties:")
        print(df["medical_specialty"].nunique())

        print("\nTop 20 specialties by count:")
        print(df["medical_specialty"].value_counts().head(20))