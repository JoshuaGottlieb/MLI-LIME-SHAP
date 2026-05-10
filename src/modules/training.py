import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Custom functions for saving and loading files
from modules.io_utils import load_object, save_object

def fit_model(
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, pd.DataFrame],
    model_name: str,
    model: BaseEstimator,
    scoring: str,
    grid_search: Optional[bool] = False,
    param_grid: Optional[Dict[str, Any]] = {},
    cv: Optional[int] = 5,
    random_state: Optional[int] = 42,
    save: Optional[bool] = True,
    save_path: Optional[str] = "./models/temp_model.pickle",
    compression: Optional[str] = None,
) -> BaseEstimator:
    """
    Fit a scikit-learn model, optionally perform GridSearchCV, and save the final estimator.

    This function builds a pipeline containing the user-provided estimator, performs
    hyperparameter tuning if requested, fits the model on the training data, and returns
    the **final fitted estimator** (not the Pipeline or GridSearchCV wrapper).

    Args:
        X_train (pd.DataFrame):
            Training feature matrix.
        y_train (Union[pd.Series, pd.DataFrame]):
            Training labels or target values.
        model_name (str):
            Name used for the model step inside the pipeline (e.g., "clf").
        model (BaseEstimator):
            Any scikit-learn compatible estimator (e.g., `RandomForestClassifier`,
            `LogisticRegression`, etc.).
        scoring (str):
            Metric used for GridSearchCV refitting (e.g., "accuracy", "neg_mse").
        grid_search (bool, optional):
            Whether to perform hyperparameter tuning via GridSearchCV.
            Defaults to False.
        param_grid (Dict[str, Any], optional):
            Dictionary of hyperparameters for GridSearchCV. Keys must follow the
            pipeline format: `"{model_name}__param_name": [...]`.
            Defaults to an empty dictionary.
        cv (int, optional):
            Number of cross-validation folds used in GridSearchCV.
            Defaults to 5.
        random_state (int, optional):
            Random seed for reproducibility. Passed to models that expose this parameter.
            Defaults to 42.
        save (bool, optional):
            Whether to save the final fitted estimator as a pickle file.
            Defaults to True.
        save_path (str, optional):
            File path where the model will be saved.
            Defaults to `"./models/temp_model.pickle"`.
        compression (str, optional):
            Compression format for saving the model (`"gzip"`, `"bz2"`, `"lzma"`, or None).
            Defaults to None.

    Returns:
        BaseEstimator:
            The final fitted estimator. If GridSearchCV is used, this is the
            best estimator extracted from the grid search; otherwise, it is
            the original model fitted on the data.
    """
        
    # GridSearchCV if requested
    if grid_search:
        # Define the pipeline with the given model
        pipeline = Pipeline(steps = [(model_name, clone(model))])
        
        full_model = GridSearchCV(
            estimator = pipeline,
            param_grid = param_grid,
            cv = cv,
            scoring = scoring,
            refit = scoring,
            n_jobs = -1,
            verbose = 1
        )
    else:
        full_model = model
        
    # Fit the model on the training data
    full_model.fit(X_train, y_train)

    if grid_search:
        full_model = full_model.best_estimator_[-1]
    
    # Save the trained model if requested
    if save:
        save_object(full_model, save_path, compression = compression)
    
    return full_model
    
def get_model_predictions_and_scores(
    X: pd.DataFrame,
    y: pd.Series,
    model_dir: str,
    metrics_dir: str,
    model_types: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Load trained models, generate predictions/probabilities, compute evaluation metrics,
    and save results to disk.

    Args:
        X (pd.DataFrame): Feature matrix for prediction.
        y (pd.Series): Ground-truth labels.
        model_dir (str): Directory containing saved model files.
        metrics_dir (str): Directory where prediction and metric files will be saved.
        model_types (List[str]): List of model type names (prefixes of pickle file names).

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing:
            - 'predictions': DataFrame of predictions and probabilities.
            - 'metrics': DataFrame of evaluation scores per model.
    """

    predictions = {}
    metrics = {}

    # Loop over all model types to load, predict, and score
    for model_type in model_types:
        model_path = os.path.join(model_dir, f"{model_type}.pickle.xz")
        model = load_object(model_path)

        # Generate predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # Compute metrics
        metrics[model_type] = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "prauc": average_precision_score(y, y_prob),
        }

        # Store prediction outputs
        predictions[f"{model_type}-predictions"] = y_pred
        predictions[f"{model_type}-probabilities"] = y_prob

    # Add ground truth last
    predictions["ground_truth"] = y.values

    # Convert dictionaries to DataFrames
    pred_df = pd.DataFrame(predictions)
    metrics_df = pd.DataFrame(metrics).reset_index(names = "metric")

    # Ensure output directory exists
    os.makedirs(metrics_dir, exist_ok = True)

    # Save to disk
    pred_df.to_csv(os.path.join(metrics_dir, "model_predictions.csv"), index = False)
    metrics_df.to_csv(os.path.join(metrics_dir, "model_metrics.csv"), index = False)

    return {"predictions": pred_df, "metrics": metrics_df}