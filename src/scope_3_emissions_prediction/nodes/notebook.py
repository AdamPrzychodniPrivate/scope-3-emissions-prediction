def _remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to remove all rows with missing values in a pandas dataframe.

    Args:
        df (pd.DataFrame): Input pandas DataFrame

    Returns:
        pd.DataFrame: Output DataFrame with rows containing missing values removed.
    """

    df_cleaned = df.dropna()

    return df_cleaned


def preprocess_scope3(scope3_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Preprocesses the Scope 3 data.

    Args:
        scope3_data: Raw data.

    Returns:
        Preprocessed data, with missing values removed.
    """

    features = scope3_data[parameters["features"]]
    preprocessed_data = _remove_missing_values(features)
    return preprocessed_data

