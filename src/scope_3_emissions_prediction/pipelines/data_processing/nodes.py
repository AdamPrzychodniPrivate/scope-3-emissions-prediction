import pandas as pd
from typing import Dict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# def _remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
#     df_cleaned = df.dropna()
#
#     return df_cleaned


from typing import List, Union


def _remove_rows_with_missing_values(df: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Remove all rows containing missing values either from the whole DataFrame or from specific columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (Union[str, List[str]], optional): Column or list of columns to consider for row removal.
                                                   If None, consider all columns. Default is None.

    Returns:
        pd.DataFrame: DataFrame with rows containing missing values removed.
    """

    if columns is not None:
        return df.dropna(subset=columns)
    else:
        return df.dropna()


from sklearn.ensemble import IsolationForest


def _remove_outliers_isolation_forest(df: pd.DataFrame, contamination: float = 0.2) -> pd.DataFrame:
    """
    Remove outliers using the Isolation Forest algorithm.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical columns.
        contamination (float): Proportion of outliers in the dataset.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """

    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Initialize the IsolationForest model
    clf = IsolationForest(contamination=contamination)

    # Fit the model on numerical columns
    clf.fit(df[numerical_cols])

    # Get outlier predictions
    outlier_predictions = clf.predict(df[numerical_cols])

    # Remove outliers from the original DataFrame based on the predictions
    df_filtered = df[outlier_predictions == 1]

    return df_filtered


# def preprocess_scope3(scope3_data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
#     """Preprocesses the Scope 3 data.
#
#     Args:
#         scope3_data: Raw data.
#
#     Returns:
#         Preprocessed data, with missing values removed.
#     """
#
#     features = scope3_data[parameters["features"]]
#     preprocessed_data = _remove_missing_values(features)
#     return preprocessed_data

def preprocess_data(data: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Preprocesses data.

    Args:
        data: Raw data.

    Returns:
        Preprocessed data, with missing values removed.
    """
    # Example
    df = data[parameters["features"]]
    df = _remove_rows_with_missing_values(df)
    df = _remove_outliers_isolation_forest(df)
    preprocessed_data = df

    return preprocessed_data


# def _outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
#     # Identify numerical columns
#     numerical_cols = df.select_dtypes(include=['number']).columns
#
#     # Initialize the IsolationForest model
#     clf = IsolationForest(contamination=0.2)  # contamination: proportion of outliers in the data set
#
#     # Fit the model on numerical columns
#     clf.fit(df[numerical_cols])
#
#     # Get outlier predictions
#     outlier_predictions = clf.predict(df[numerical_cols])
#
#     # Remove outliers from the original DataFrame based on the predictions
#     df_filtered = df[outlier_predictions == 1]
#
#     return df_filtered


def _remap_industry(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    industries_to_keep = df['Industry (Exiobase)'].value_counts()[df['Industry (Exiobase)'].value_counts() > 50].index
    df['Industry (Exiobase)'] = df['Industry (Exiobase)'].apply(lambda x: x if x in industries_to_keep else 'Other')
    return df


def _create_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    interaction_pairs = [
        ('Use of Sold Products', 'Processing of Sold Products'),
        ('Use of Sold Products', 'Purchased Goods and Services'),
        ('Processing of Sold Products', 'Purchased Goods and Services'),
        ('Purchased Goods and Services', 'End of Life Treatment of Sold Products')
    ]

    for col1, col2 in interaction_pairs:
        new_col_name = f"{col1}_x_{col2}"
        df[new_col_name] = df[col1] * df[col2]

    return df


def _create_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_square = [
        'Use of Sold Products',
        'Processing of Sold Products',
        'Purchased Goods and Services',
        'End of Life Treatment of Sold Products'
    ]

    for col in cols_to_square:
        new_col_name = f"{col}_Squared"
        df[new_col_name] = df[col] ** 2

    return df


def _one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encode 'Country' and 'Industry (Exiobase)' columns
    df_encoded = pd.get_dummies(df, columns=['Industry (Exiobase)'])
    return df_encoded


def _normalization(df: pd.DataFrame) -> pd.DataFrame:
    # Create the scaler
    scaler_standard = StandardScaler()

    # Fit the scaler to the data (excluding categorical data if not already encoded)
    df_normalized_standard = pd.DataFrame(scaler_standard.fit_transform(df), columns=df.columns)

    return df_normalized_standard


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conducts feature engineering on the given DataFrame.

    Steps:
    1. Outlier Removal: Removes outliers using the Isolation Forest algorithm.
    2. Remap Industry: Aggregates less frequent industry categories into 'Other'.
    3. Create Interaction Terms: Creates new features by multiplying pairs of existing features.
    4. Create Polynomial Features: Squares selected features to create new polynomial features.
    5. One-Hot Encoding: One-hot encodes categorical features.
    6. Normalization: Standardizes the feature values.

    Args:
        df: Original DataFrame.

    Returns:
        df_feature_engineered: DataFrame after feature engineering.
    """

    # df = _outlier_removal(df)
    df = _remap_industry(df)
    df = _create_interaction_terms(df)
    df = _create_polynomial_features(df)
    df = _one_hot_encode(df)
    df = _normalization(df)
    df_feature_engineered = df

    return df_feature_engineered