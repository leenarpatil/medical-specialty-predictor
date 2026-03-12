def clean_dataset(df):
    """
    Clean the healthcare transcription dataset.

    Steps:
    1. Drop unnecessary columns
    2. Remove rows with missing target or text
    3. Remove duplicates
    4. Strip whitespace
    5. Remove empty transcription rows
    6. Reset index

    Parameters:
        df (pd.DataFrame): Raw dataframe

    Returns:
        pd.DataFrame: Cleaned dataframe
    """

    data = df.copy()

    # Drop unnecessary columns if they exist
    cols_to_drop = ["Unnamed: 0", "keywords"]
    for col in cols_to_drop:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    # Drop rows where target or main text is missing
    data = data.dropna(subset=["medical_specialty", "transcription"])

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Strip spaces from text columns
    text_cols = ["description", "sample_name", "transcription", "medical_specialty"]
    for col in text_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # Remove empty transcription rows
    data = data[data["transcription"] != ""]

    # Reset index
    data = data.reset_index(drop=True)

    return data


def inspect_cleaned_data(data):
    """
    Print basic information about cleaned dataset.

    Parameters:
        data (pd.DataFrame): Cleaned dataframe
    """
    print("\n========== CLEANED DATASET INSPECTION ==========")
    print("Shape after cleaning:", data.shape)

    print("\nRemaining columns:")
    print(data.columns.tolist())

    print("\nMissing values after cleaning:")
    print(data.isnull().sum())

    print("\nNumber of unique specialties after cleaning:")
    print(data["medical_specialty"].nunique())

    print("\nTop 20 specialties after cleaning:")
    print(data["medical_specialty"].value_counts().head(20))