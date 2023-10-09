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
    
    df = scope3_data[parameters["features"]]
    df = _remove_missing_values(df)
    preprocessed_data = df
    
    return preprocessed_data

def _outlier_removal(df: pd.DataFrame) -> pd.DataFrame:
    # Identify numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Initialize the IsolationForest model
    clf = IsolationForest(contamination=0.2)  # contamination: proportion of outliers in the data set

    # Fit the model on numerical columns
    clf.fit(df[numerical_cols])

    # Get outlier predictions
    outlier_predictions = clf.predict(df[numerical_cols])

    # Remove outliers from the original DataFrame based on the predictions
    df_filtered = df[outlier_predictions == 1]

    return df_filtered
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
    df = _outlier_removal(df)
    df = _remap_industry(df)
    df = _create_interaction_terms(df)
    df = _create_polynomial_features(df)
    df = _one_hot_encode(df)
    df = _normalization(df)
    df_feature_engineered = df
    
    return df_feature_engineered

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def split_data(data: pd.DataFrame, model_options: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    # X = data[parameters["features"]]
    X = data[parameters["features"]].drop("Scope 3", axis=1)
    y = data["Scope 3"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import logging
import pandas as pd
from typing import Any

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Trains the XGBoost model.
    
    Args:
        X_train: Training data of independent features.
        y_train: Training data for target variable.
        
    Returns:
        Trained model.
    """
    params = {
        'alpha': 9.418025790529975e-05,
        'colsample_bytree': 0.73850137825373,
        'eta': 0.03756810920990241,
        'gamma': 1.8103086083962833e-05,
        'lambda': 0.006052853661670603,
        'max_depth': 4,
        'min_child_weight': 1.0000000000000004e-06,
        'objective': 'reg:squarederror',
        'subsample': 0.8954379516782436,
        'eval_metric': ['rmse', 'mae']
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, dtrain, num_boost_round=674)
    return model


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series):
    """Calculates and logs the coefficient of determination and RMSE.
    
    Args:
        model: Trained XGBoost model.
        X_test: Testing data of independent features.
        y_test: Testing data for target variable.
    """
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred = model.predict(dtest)
    score = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    print(f"Model has a coefficient R^2 of {score:.3f} on test data.")
    print(f"Model has a RMSE of {rmse:.3f} on test data.")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Model has a coefficient R^2 of {score:.3f} on test data.")
    logger.info(f"Model has a RMSE of {rmse:.3f} on test data.")
