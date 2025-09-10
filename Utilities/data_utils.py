import pandas as pd
from typing import List


def prepare_delta_features(feature_data: pd.DataFrame, price_columns: List[str] = ['close', 'open', 'high', 'low']) -> pd.DataFrame:
    """
    Calculate delta features for price-related columns in the DataFrame.
    
    Args:
        feature_data: Input DataFrame containing features
        price_columns: List of column names to calculate deltas for (default: ['close', 'open', 'high', 'low'])
    
    Returns:
        DataFrame with delta features, with the first row (containing NaN values) dropped
    """
    # Create a copy of the input DataFrame
    feature_data_copy = feature_data.copy()
    
    # Iterate through the price_columns and calculate deltas
    for column in price_columns:
        if column in feature_data_copy.columns:
            feature_data_copy[column] = feature_data_copy[column].diff()
    
    # Return the transformed DataFrame after dropping the first row that contains NaN values
    return feature_data_copy.iloc[1:]