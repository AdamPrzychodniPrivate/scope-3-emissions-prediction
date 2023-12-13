import logging
from typing import Dict, Tuple

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def split_data(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}


import logging
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import logging
import pandas as pd
from typing import Any


def split_data_scope3(data: pd.DataFrame, parameters: Dict) -> Tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    X = data[parameters["features"]]
    # X = data[parameters["features"]].drop("Scope 3", axis=1)
    y = data[parameters["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model_scope3(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
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


def evaluate_model_scope3(model: Any, X_test: pd.DataFrame, y_test: pd.Series):
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

    logger = logging.getLogger(__name__)
    logger.info(f"Model has a coefficient R^2 of {score:.3f} on test data.")
    logger.info(f"Model has a RMSE of {rmse:.3f} on test data.")