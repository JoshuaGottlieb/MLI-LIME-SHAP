from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats.contingency import association

def calculate_VIF(
    dataframe: pd.DataFrame,
    columns: Optional[List[str]] = None,
    log1p_columns: Optional[List[str]] = None,
    ridge: float = 1e-8,
    verbose: bool = False
) -> pd.Series:
    """
    Calculate Variance Inflation Factors (VIF) for numeric columns in a DataFrame.

    VIF is calculated as the diagonal of the inverse of the correlation matrix:
        VIF_j = diag(inv(corr_matrix))

    This version handles singular matrices by adding a small ridge term.

    Args:
        dataframe (pd.DataFrame): Input DataFrame.
        columns (List[str], optional): Subset of columns to compute VIF for.
            Defaults to all numeric columns.
        log1p_columns (List[str], optional): List of columns to apply np.log1p transformation
            before computing correlation. Defaults to None.
        ridge (float, optional): Small value to add to diagonal to handle singular matrices. Defaults to 1e-8.
        verbose (bool, optional): If True, print warnings for high VIF (>10). Defaults to False.

    Returns:
        pd.Series: VIF values indexed by column names, series name "VIF".
    """
    
    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include = np.number).columns.tolist()
    
    if columns is not None:
        numeric_cols = [col for col in columns if col in numeric_cols]

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for VIF calculation.")

    # Copy and transform log1p columns if specified
    df_vif = dataframe[numeric_cols].copy()
    if log1p_columns:
        for col in log1p_columns:
            if col in df_vif.columns:
                df_vif[col] = np.log1p(df_vif[col])
    
    # Compute correlation matrix
    corr_matrix = df_vif.corr(numeric_only = True)
    
    # Regularize diagonal to handle singular matrices
    corr_matrix += np.eye(len(corr_matrix)) * ridge

    # Invert correlation matrix
    inv_corr_matrix = np.linalg.inv(corr_matrix.values)
    
    # VIF is the diagonal of the inverse correlation matrix
    vif_values = pd.Series(np.diag(inv_corr_matrix), index = corr_matrix.columns, name = "VIF")
    
    # Verbose warnings for high VIF
    if verbose:
        high_vif = vif_values[vif_values > 10]
        if not high_vif.empty:
            print("Warning: High multicollinearity detected. Variables with VIF > 10:")
            for col, val in high_vif.items():
                print(f"  - {col}: {val:.2f}")
    
    return vif_values

def compute_pairwise_associations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise association metrics (Cramer's V) between all categorical columns in a DataFrame.

    Args:
        dataframe (pd.DataFrame): 
            Input DataFrame containing categorical variables.

    Returns:
        pd.DataFrame:
            A DataFrame with one row per unique column pair and four columns:
            - 'column1': The first variable in the pair.
            - 'column2': The second variable in the pair.
            - 'cramers_v': Cramer's V statistic for association strength.
    """
    
    associations = []
    
    # Loop through all unique pairs of columns
    for i, col1 in enumerate(dataframe):
        for col2 in dataframe[i + 1:]:
            # Create contingency table for the two categorical variables
            crosstab = pd.crosstab(dataframe[col1], dataframe[col2])
            
            # Compute Cramer's V
            # Use correction only if the table is exactly 2x2
            cramers_v = association(crosstab, correction = crosstab.shape == (2, 2))
            
            # Store results for this column pair
            associations.append((col1, col2, cramers_v))

    # Convert results into a DataFrame
    associations_df = pd.DataFrame(
        associations,
        columns = ['column1', 'column2', 'cramers_v']
    )

    return associations_df