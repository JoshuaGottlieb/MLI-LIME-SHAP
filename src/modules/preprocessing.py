from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def convert_column_names_and_types(
    df: pd.DataFrame,
    target: str,
    int_columns: List[str],
    binary_columns: List[str]
) -> pd.DataFrame:
    """
    Rename the target column to 'death' and convert column types:
    - integer columns → int
    - binary columns → category

    Args:
        df (pd.DataFrame): Input dataframe.
        target (str): Name of the target column to rename to 'death'.
        int_columns (List[str]): Columns to convert to integer dtype.
        binary_columns (List[str]): Columns to convert to categorical dtype.

    Returns:
        pd.DataFrame: Updated dataframe with renamed columns and converted types.
    """

    # Rename target column
    df = df.rename({target: "death"}, axis = 1)

    # Convert integer columns
    for col in int_columns:
        df[col] = df[col].astype(int)

    # Convert binary/categorical columns
    for col in binary_columns:
        df[col] = df[col].astype("category")

    return df

def encode_features(
    X: pd.DataFrame,
    y: pd.DataFrame,
    numeric_cols: List[str],
    passthrough_cols: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """
    Split the dataset into training and testing subsets and apply column-wise
    preprocessing transformations.

    This function performs a stratified train/test split using ``y`` and fits
    a ``ColumnTransformer`` on the training features. The preprocessing pipeline
    includes:
        - Standard scaling of numeric columns
        - Passthrough of selected columns without modification

    The fitted transformer is returned to enable consistent transformation
    or inverse transformation in downstream workflows.

    Args:
        X (pd.DataFrame):
            DataFrame containing all feature columns.
        y (pd.DataFrame):
            Target or label data used for stratified splitting and later
            evaluation. It is not used as a supervised target during
            preprocessing.
        numeric_cols (List[str]):
            Names of numeric columns to scale using ``StandardScaler``.
        passthrough_cols (List[str]):
            Names of columns to include in the output without transformation.
        test_size (float, optional):
            Proportion of the dataset reserved for the test split.
            Defaults to 0.2.
        random_state (int, optional):
            Random seed for reproducibility of the train/test split.
            Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
            A tuple ``(train, test, preprocessor)`` where:
                - ``train`` is the transformed training DataFrame with ``y`` appended
                - ``test`` is the transformed test DataFrame with ``y`` appended
                - ``preprocessor`` is the fitted ``ColumnTransformer`` used to
                  transform the feature data
    """

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state,
        stratify = y
    )

    # Column-specific preprocessing: scaling, encoding, passthrough
    preprocessor = ColumnTransformer(
        transformers = [
            ('ssc', StandardScaler(), numeric_cols),                # Scale numeric features
            ('pass', 'passthrough', passthrough_cols)               # Keep passthrough features unchanged
        ],
        sparse_threshold = 0,               # Force dense numpy arrays
        n_jobs = -1,                        # Parallel processing
        verbose_feature_names_out = False   # Cleaner feature names
    )

    # Fit on training set and transform both training and test data
    X_train_tr = preprocessor.fit_transform(X_train, y_train)
    X_train_tr = pd.DataFrame(
        X_train_tr,
        index = X_train.index,
        columns = preprocessor.get_feature_names_out()
    )

    X_test_tr = preprocessor.transform(X_test)
    X_test_tr = pd.DataFrame(
        X_test_tr,
        index = X_test.index,
        columns = preprocessor.get_feature_names_out()
    )

    # Reattach y for evaluation
    train = pd.concat([X_train_tr, y_train], axis = 1)
    test = pd.concat([X_test_tr, y_test], axis = 1)

    return train, test, preprocessor

def decode_data(
    X: pd.DataFrame,
    encoder: ColumnTransformer
) -> pd.DataFrame:
    """
    Decode a transformed feature matrix back to its original feature scale.

    This function reverses the transformation applied by a fitted
    ``ColumnTransformer`` by applying the inverse transformation of the
    standard-scaling component (assumed to be named ``'ssc'``). It then
    reattaches pass-through features and casts selected columns back to
    integer type for readability and semantic correctness.

    The function assumes:
    - The final 5 columns of ``X`` were passed through unchanged.
    - A transformer named ``'ssc'`` exists in the encoder.
    - The order of output features matches ``encoder.get_feature_names_out()``.

    Args:
        X (pd.DataFrame):
            Transformed feature matrix produced by the fitted encoder.
        encoder (ColumnTransformer):
            Fitted ``ColumnTransformer`` containing a standard-scaling
            transformer under the key ``'ssc'``.

    Returns:
        pd.DataFrame:
            Decoded DataFrame with original feature scales restored and
            pass-through features appended.
    """

    # Separate scaled features from pass-through features
    X_ssc = X.iloc[:, :-5]
    X_pt = X.iloc[:, -5:]

    # Apply inverse scaling to recover original feature values
    X_unscaled = pd.DataFrame(
        encoder.named_transformers_["ssc"].inverse_transform(X_ssc),
        index = X_ssc.index,
        columns = encoder.get_feature_names_out()[:-5]
    )

    # Columns that should be represented as integers after decoding
    int_columns = [
        "age",
        "creatinine_phosphokinase",
        "ejection_fraction",
        "platelets",
        "serum_sodium",
        "time",
    ]

    # Cast selected columns back to integer type
    for col in int_columns:
        if col in X_unscaled.columns:
            X_unscaled[col] = X_unscaled[col].astype(int)

    # Reattach pass-through features
    X_decoded = pd.concat([X_unscaled, X_pt], axis = 1)

    return X_decoded