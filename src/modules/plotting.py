import os
import re
from typing import List, Optional, Tuple, Union

import eli5
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from matplotlib.axes import Axes
from pandas.io.formats.style import Styler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Custom utility functions for formatting plots
from modules.plotting_utils import (
    snake_to_title,
    snake_to_title_axes,
    snake_to_title_ticks,
)

# Custom functions for loading and saving files
from modules.io_utils import load_object

# Custom functions for statistics
from modules.statistics import compute_pairwise_associations

# ---- Correlation / Association Plots ----

def correlation_heatmap(
    dataframe: pd.DataFrame,
    figsize: tuple[float, float] = (12, 6),
    abs: bool = True
) -> plt.Figure:
    """
    Draw a Pearson correlation heatmap with title-cased axis labels.

    Args:
        dataframe (pd.DataFrame):
            Input dataframe containing numeric columns to correlate.
        figsize (tuple, optional):
            Figure size if axis is created. Defaults to (12, 6).
        abs (bool, optional):
            If True, take the absolute value of correlations.
            If False, keep signed correlations. Defaults to True.

    Returns:
        matplotlib.figure.Figure:
            Figure containing the Pearson correlation heatmap.
    """
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize = figsize)

    # Compute Pearson correlations
    corr = dataframe.corr(method = 'pearson', numeric_only = True)

    # Optionally take absolute value
    if abs:
        corr = corr.abs()

    # Draw heatmap
    sns.heatmap(corr, annot = True, cbar = False, ax = ax, fmt = ".2f")

    # Title
    ax.set_title("Pearson Correlation")

    # Format tick labels to title case
    snake_to_title_ticks(ax, y = True, rotation_x = 45, rotation_y = 0)

    plt.tight_layout()

    return fig

def association_heatmap(
    dataframe: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 6)
) -> Axes:
    """
    Generate a heatmap of pairwise Cramer's V associations between
    categorical variables.

    This function computes pairwise Cramer's V values using
    `compute_pairwise_associations()`, reshapes the results into a
    square matrix, and draws a heatmap. Axis tick labels are converted
    to title case using `snake_to_title_ticks()`.

    Args:
        dataframe (pd.DataFrame):
            Input DataFrame containing categorical variables.
        figsize (Tuple[int, int], optional):
            Figure size for the heatmap. Defaults to (12, 6).

    Returns:
        matplotlib.axes.Axes:
            The Axes object containing the Cramer's V heatmap.
    """

    # Create a single subplot for the Cramer's V heatmap
    fig, ax = plt.subplots(figsize = figsize)

    # Compute pairwise association statistics in long format
    associations_df = compute_pairwise_associations(dataframe)

    # Pivot the Cramer's V values into a square matrix
    cramers_v = associations_df.pivot(
        index = "column1",
        columns = "column2",
        values = "cramers_v"
    )
    cramers_v.index.name = None
    cramers_v.columns.name = None

    # Draw Cramer's V heatmap
    sns.heatmap(cramers_v, annot = True, fmt = ".2g", cbar = False, ax = ax)

    # Apply title-case labels
    snake_to_title_ticks(ax, rotation_x = 45, rotation_y = 0)

    # Set plot title
    ax.set_title("Cramer's V")

    plt.tight_layout()

    return ax

# ---- Metrics and Predictions ----

def plot_confusion_matrix(
    prediction_frame: pd.DataFrame,
    model: str,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (6, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a confusion matrix heatmap for a given model's predictions.

    Args:
        prediction_frame (pd.DataFrame): DataFrame containing model predictions and ground truth.
            Must contain columns:
                - f"{model}-predictions"
                - "ground_truth"
        model (str): Model identifier prefix (e.g., "logreg", "rf").
        class_names (Optional[List[str]]): Names of the classes in order.
            Defaults to ["Survived", "Died"] if None.
        figsize (tuple): Size of the figure if a new one is created.
        ax (Optional[plt.Axes]): Axis object to draw on. If None, a new axis is created.

    Returns:
        plt.Axes: The matplotlib Axes containing the heatmap.
    """

    # Default class labels
    if class_names is None:
        class_names = ["Survived", "Died"]

    # Extract predictions and true values
    y_pred = prediction_frame[f"{model}-predictions"]
    y_true = prediction_frame["ground_truth"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # If an axis is not provided, create one
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    # Draw heatmap
    sns.heatmap(
        cm,
        annot = True,
        fmt = "d",
        cmap = "Blues",
        cbar = False,
        xticklabels = class_names,
        yticklabels = class_names,
        ax = ax
    )

    # Axis labels and title
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title(f"Confusion Matrix: {model.upper()}")

    return ax

def style_dataframe(
    df: pd.DataFrame,
    stripe_light: str = "white",
    stripe_dark: str = "#f2f2f2",
    text_color: str = "#222",
    max_bg: str = "#c7dbf4",
    cell_width: Optional[Union[int, str]] = "120px",
) -> pd.io.formats.style.Styler:
    """
    Style a pandas DataFrame:
        - Hide index
        - Center text with dark font color
        - Alternating row striping
        - Highlight per-column maxima using background color
        - Bold columns that are fully non-numeric
        - Control cell width

    Args:
        df: DataFrame to style.
        stripe_light: Background color for even rows.
        stripe_dark: Background color for odd rows.
        text_color: Text color (dark recommended).
        max_bg: Background color for max values.
        cell_width: Width for each cell (e.g., '120px').

    Returns:
        pandas.io.formats.style.Styler
    """

    # Determine which columns are fully non-numeric
    non_numeric_cols = [
        col for col in df.columns
        if pd.to_numeric(df[col], errors = "coerce").isna().all()
    ]

    # Map index -> row number
    index_to_pos = {idx: i for i, idx in enumerate(df.index)}

    # Row striping
    def stripe_rows(row):
        pos = index_to_pos[row.name]
        bg = stripe_dark if pos % 2 else stripe_light
        return [f"background-color: {bg}; color: {text_color}" for _ in row]

    # Highlight max values
    def highlight_max(col):
        numeric = pd.to_numeric(col, errors = "coerce")
        is_max = numeric == numeric.max()

        styles = []
        for flag in is_max:
            if flag:
                styles.append(
                    f"background-color: {max_bg}; font-weight: bold; color: {text_color}"
                )
            else:
                styles.append(f"color: {text_color}")
        return styles

    # Bold fully non-numeric columns
    def bold_non_numeric(col):
        if col.name in non_numeric_cols:
            return ["font-weight: bold;"] * len(col)
        else:
            return [""] * len(col)

    # Start styling
    styler = df.style

    # Hide index
    styler = styler.hide(axis = "index")

    styler = (
        styler
        .apply(stripe_rows, axis = 1)         # row striping
        .apply(highlight_max, axis = 1)       # background highlight for max values
        .apply(bold_non_numeric, axis = 0)    # bold non-numeric columns
        .set_properties(
            **{
                "text-align": "center",
                "color": text_color,
                "width": cell_width,
            }
        )
        .set_table_styles(
            [
                # Header cells
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#eaeaea"),
                        ("text-align", "center"),
                        ("color", text_color),
                        ("border", "1px solid #ccc"),
                        ("font-weight", "bold"),
                    ],
                },
                # Data cells
                {
                    "selector": "td",
                    "props": [
                        ("border", "1px solid #ccc"),
                        ("width", cell_width),
                    ],
                },
                # Table container
                {
                    "selector": "table",
                    "props": [("border-collapse", "collapse")],
                },
            ]
        )
    )

    return styler

def style_model_metrics(metrics: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Clean, relabel, and style a model metrics table using `style_dataframe`.

    Steps:
        - Replace internal metric codes with human-readable names
        - Rename columns to readable model names
        - Apply `style_dataframe` styling (striping, centering, max highlighting, etc.)

    Args:
        metrics (pd.DataFrame): A DataFrame with columns ['metric', model_1, model_2, ...].

    Returns:
        pd.io.formats.style.Styler: Styled version of the metrics table.
    """

    # Rename metric values
    metrics = metrics.copy()
    metrics["metric"] = metrics["metric"].replace(
        {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1 Score",
            "prauc": "Average Precision (PR AUC)",
        }
    )

    # Rename columns (first column becomes blank header)
    metrics.columns = [
        "",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
    ]

    # Apply existing dataframe styling function
    return style_dataframe(metrics, cell_width = "200px")
    
# ---- Explainability Graphs ----

def eli5_global_feature_plot(
    model_type: str,
    model_dir: str,
    feature_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Generate a global feature-importance or coefficient plot using ELI5 for a saved model.

    This function loads a serialized model (saved as ``{model_type}.pickle.xz`` inside
    ``model_dir``), extracts its global feature weights via ``eli5.explain_weights_df``,
    and produces a bar plot showing the magnitude of each non-zero weight.

    Args:
        model_type (str):
            A short identifier for the model (e.g., ``"dt"``, ``"rf"``, ``"logreg"``).
            This determines both which file to load and how the y-axis is labeled.
        model_dir (str):
            Directory containing the saved model file. The expected filename is
            ``"{model_type}.pickle.xz"``.
        feature_names (List[str]):
            Ordered list of feature names that correspond to the model's input matrix.
            These are passed to ELI5 to map coefficients to human-readable labels.
        figsize (Tuple[int, int], optional):
            Size of the figure used when ``ax`` is not provided.
            Defaults to ``(12, 6)``.
        ax (matplotlib.axes.Axes, optional):
            An existing axis to draw the bar plot on. If ``None``, a new figure
            and axis will be created.

    Returns:
        matplotlib.axes.Axes:
            The axis containing the generated bar plot.

    Notes:
        - For linear models such as logistic regression, the plot shows signed weights.
        - For tree-based models, the plot shows feature importances.
        - Zero-valued weights are excluded from the plot for clarity.
    """

    # Load the trained model from disk
    model_path = os.path.join(model_dir, f"{model_type}.pickle.xz")
    model = load_object(model_path)

    # Create a new axis if one was not provided
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    # Extract feature weights/importance using ELI5
    weights: pd.DataFrame = eli5.explain_weights_df(model, feature_names = feature_names)

    # Remove zero-weight features for cleaner visualization
    weights = weights[weights["weight"] != 0]

    # Plot the bar chart of weights
    sns.barplot(data=weights, x = "feature", y = "weight", ax = ax)

    # Add numeric labels above bars
    for container in ax.containers:
        ax.bar_label(container, fmt = "{:.4f}")

    # Choose label text based on model type
    weight_type = "Weights" if model_type == "logreg" else "Feature Importances"

    # Axis labeling and formatting
    ax.set_xlabel("Feature")
    ax.set_ylabel(weight_type)

    # Beautify x-tick labels
    ax.set_xticks(ax.get_xticks()) # Called to suppress matplotlib warning
    ax.set_xticklabels(
        [feat.replace("_", " ").title() for feat in weights["feature"].tolist()]
    )

    ax.set_title(f"ELI5 Model {weight_type} for {model_type.upper()}")

    return ax

def eli5_local_prediction_explanations(
    model_type: str,
    model_dir: str,
    X_test: pd.DataFrame,
    model_idx: List[int],
    idx_types: List[str] = ['True Positive', 'True Negative', 'False Positive', 'False Negative'],
    figsize: Tuple[int, int] = (12, 6),
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Generate local feature importance explanations for selected test instances using ELI5.

    This function loads a saved model, computes ELI5 explanations for the specified
    rows in the test set, and produces a bar plot comparing feature contributions
    for each instance.

    Args:
        model_type (str):
            Name of the model (e.g., 'logreg', 'dt') used for labeling and determining
            whether to display weights or feature importances.
        model_dir (str):
            Directory containing the saved model file (expected as "{model_type}.pickle.xz").
        X_test (pd.DataFrame):
            Test features DataFrame containing the instances to explain.
        model_idx (List[int]):
            List of row indices from X_test for which to generate local explanations.
        idx_types (List[str], optional):
            List of labels corresponding to each explained instance
            (e.g., True Positive, True Negative). Defaults to four common categories.
        figsize (Tuple[int, int], optional):
            Size of the figure if a new axis is created. Defaults to (12, 6).
        ax (matplotlib.axes.Axes, optional):
            Existing axis to draw the bar plot. If None, a new figure and axis are created.

    Returns:
        matplotlib.axes.Axes:
            The axis containing the bar plot of local feature contributions.
    """

    # Load the trained model from disk
    model_path = os.path.join(model_dir, f"{model_type}.pickle.xz")
    model = load_object(model_path)

    # Initialize DataFrame to collect explanations
    prediction_df = pd.DataFrame()

    # Compute ELI5 explanations for each selected instance
    for idx, idx_type in zip(model_idx, idx_types):
        pred_exp = eli5.explain_prediction_df(model, X_test.iloc[idx])
        pred_exp['type'] = idx_type
        prediction_df = pd.concat([prediction_df, pred_exp], ignore_index = True)

    # Create a new axis if none is provided
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    # Plot feature weights for each instance type
    sns.barplot(data = prediction_df, x = 'feature', y = 'weight', hue = 'type', ax = ax)

    # Add numeric labels above bars
    for container in ax.containers:
        ax.bar_label(container, fmt = '{:.2f}')

    # Determine label for y-axis based on model type
    weight_type = "Weights" if model_type == "logreg" else "Feature Importances"
    ax.set_xlabel("Feature")
    ax.set_ylabel(weight_type)

    # Beautify x-tick labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [feat.replace('_', ' ').title() for feat in prediction_df["feature"].unique().tolist()]
    )

    # Set title and format legend
    ax.set_title(f"ELI5 Model {weight_type} Prediction Explanations for {model_type.upper()}")
    ax.legend(title = '')

    return ax

def plot_local_lime_explanations(
    model_type: str,
    model_dir: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_idx: List[int],
    categorical_features: List[int],
    num_features: int = 12,
    figsize: Tuple[int, int] = (10, 4)
) -> plt.Figure:
    """
    Generate LIME local explanations for selected test instances using an
    encoded model pipeline and visualize them as horizontal bar charts.

    This function loads a trained classification model, initializes a
    ``LimeTabularExplainer`` using the raw training feature matrix, and explains
    individual predictions for selected test instances. During explanation,
    inputs are transformed with the provided fitted ``ColumnTransformer``
    before being passed to the model for probability prediction.

    Each instance explanation is rendered as a horizontal bar chart showing
    feature-level contributions, the model intercept, and class probability
    context.

    Args:
        model_type (str):
            Model identifier used to load the serialized model
            (e.g., ``'rf'``, ``'dt'``).
        model_dir (str):
            Directory containing the saved model file
            ``'{model_type}.pickle.xz'``.
        X_train (pd.DataFrame):
            Raw unencoded training feature matrix used to initialize the LIME explainer.
        X_test (pd.DataFrame):
            Raw test feature matrix containing instances to be explained.
        model_idx (List[int]):
            List of row indices in ``X_test`` for which to generate local
            explanations.
        categorical_features (List[int]):
            Column indices in ``X_train`` corresponding to categorical features
            for LIME's discretization and sampling logic.
        num_features (int, optional):
            Number of features to include in each LIME explanation.
            Defaults to 12.
        figsize (Tuple[int, int], optional):
            Base figure size. The height is scaled by the number of explained
            instances. Defaults to ``(10, 4)``.

    Returns:
        plt.Figure:
            Matplotlib figure containing one horizontal bar chart per explained
            test instance, including feature contributions, intercept, and
            class probability annotations.
    """

    # Load the trained model
    model_path = os.path.join(model_dir, f"{model_type}.pickle.xz")
    model = load_object(model_path)

    # Initialize LIME explainer
    explainer = LimeTabularExplainer(
        training_data = X_train.to_numpy(),
        feature_names = X_train.columns.tolist(),
        class_names = [1, 0],
        categorical_features = categorical_features,
        mode = 'classification',
        random_state = 42
    )

    # Recreate preprocessor using indices instead of column names for use in LIME explanations
    # LIME creates NumPy arrays when generating local instances, so it cannot use named features
    encoder = ColumnTransformer(
        transformers = [
            ('ssc', StandardScaler(), [0, 1, 2, 3, 4, 5, 6]),        # Scale numeric features
            ('pass', 'passthrough', [7, 8, 9, 10, 11])               # Keep passthrough features unchanged
        ],
        sparse_threshold = 0,               # Force dense numpy arrays
        n_jobs = -1,                        # Parallel processing
        verbose_feature_names_out = False   # Cleaner feature names
    )
    encoder.fit(X_train.to_numpy())

    n_instances = len(model_idx)

    # Create figure and axes if not supplied
    fig, axes = plt.subplots(n_instances, 1, figsize = (figsize[0], figsize[1] * n_instances))
    if n_instances == 1:
        axes = np.array([axes])

    # Default labels for the type of instance
    idx_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']

    # Loop through instances and generate plots
    for i, (idx, idx_label, ax) in enumerate(zip(model_idx, idx_labels, axes)):

        # Generate LIME explanation for a single instance
        explanation = explainer.explain_instance(
            X_test.iloc[idx].to_numpy(),
            lambda x: model.predict_proba(encoder.transform(x)),
            num_features = num_features
        )
        
        # Extract explanation scores, labels, r2, intercept, and class probabilities
        exp_list = explanation.as_list()
        r2 = explanation.score
        intercept = explanation.intercept[1]
        probs = model.predict_proba(encoder.transform(X_test.iloc[[idx]].to_numpy()))[0, :]

        # Format feature names to title case and replace underscores
        def to_title_case(match):
            return match.group(0).replace('_', ' ').title()

        labels = [
            re.sub(
                r'([a-z])(=)(\d)',   # Add spaces around equals signs, ex: "smoking=0" -> "smoking = 0"
                r'\1 \2 \3',
                re.sub(
                    r'([a-z\_]+)',   # Convert snake_case to title case, ex: "serum_sodium" -> "Serum Sodium"
                    to_title_case,
                    feat[0]
                )
            )
            for feat in exp_list
        ]
        coef = [feat[1] for feat in exp_list]

        # Insert intercept at appropriate position
        intercept_idx = num_features - np.searchsorted(np.abs(np.array(coef))[::-1], intercept)
        coef.insert(intercept_idx, intercept)
        labels.insert(intercept_idx, 'Intercept')

        # Color bars: 1 = positive contribution (green), 0 = negative (red)
        colors = [1 if c > 0 else 0 for c in coef]

        # Plot horizontal bar chart
        sns.barplot(
            x = coef,
            y = labels,
            orient = 'h',
            hue = colors,
            palette = ['red', 'green'],
            ax = ax
        )

        # Add numeric labels to bars
        for container in ax.containers:
            ax.bar_label(container, fmt = '{:.2g}')

        # Axis labels and formatting
        ax.set_title(f'LIME Explanation for {model_type.upper()} | {idx_label} | R² = {r2:.2f}')
        ax.set_xlim(min(coef) - 0.05, max(coef) + 0.05)
        ax.set_xlabel('Feature Contribution')

        # Format legend with proxy artists
        red_square = mpatches.Patch(color = 'red', label = 'Class 0')
        green_square = mpatches.Patch(color = 'green', label = 'Class 1')
        handles = [red_square, green_square]
        labels = [f'{handle.get_label()}: {prob:.2f}' for handle, prob in zip(handles, probs)]
        ax.legend(handles, labels, title = 'Class Probabilities', loc = 'lower right');

    plt.tight_layout()
    
    return fig

def shap_waterfall_plot(
    shap_values,
    idx: int,
    model_name: str,
    prediction_label: str,
    max_display: int = 12,
    title_fontsize: int = 13,
) -> plt.Axes:
    """
    Generate a SHAP waterfall plot for a single observation.

    This function visualizes how individual feature contributions combine to
    produce the model output for one sample using SHAP's waterfall diagram.

    Args:
        shap_values:
            SHAP values object containing per-sample contributions. Must be
            indexable by integer position (``shap_values[idx]``).
        idx (int):
            Index of the sample to visualize.
        model_name (str):
            Name of the model producing the prediction (e.g., "XGBoost").
        prediction_label (str):
            Descriptive label for the prediction type (e.g., "True Negative").
        max_display (int, optional):
            Maximum number of SHAP components to display in the waterfall chart.
            Defaults to 12.
        title_fontsize (int, optional):
            Font size for the plot title. Defaults to 13.

    Returns:
        plt.Axes:
            The matplotlib axis containing the generated waterfall plot.
    """

    # Generate SHAP waterfall plot (suppress auto-display)
    ax = shap.plots.waterfall(
        shap_values[idx],
        max_display = max_display,
        show = False
    )

    # Add a descriptive title
    ax.set_title(
        f"Explanation for {prediction_label} Prediction ({model_name}) [Waterfall Plot]",
        fontsize = title_fontsize
    )
    
    return ax

def shap_force_plot(
    shap_values,
    X: pd.DataFrame,
    idx: int,
    model_name: str,
    prediction_label: str,
    text_rotation: int = 45,
    title_fontsize: int = 20,
    title_y: float = 1.4,
    contribution_threshold: float = 0.0,
) -> plt.Figure:
    """
    Generate a SHAP force plot for a single observation.

    This function creates a force plot showing how feature contributions push
    the prediction score higher or lower relative to the SHAP base value. It
    supports Matplotlib rendering, configurable label rotation, and optional
    filtering of small-magnitude contributions.

    Args:
        shap_values:
            SHAP values object containing feature-level contributions. Must be
            indexable by integer position (``shap_values[idx]``).
        X (pd.DataFrame):
            Feature matrix used to compute SHAP values. The sample at ``idx``
            will be displayed.
        idx (int):
            Index of the sample to visualize.
        model_name (str):
            Name of the model producing the prediction.
        prediction_label (str):
            Description of the prediction type (e.g., "True Negative").
        text_rotation (int, optional):
            Rotation angle (in degrees) for feature name labels. Defaults to 45.
        title_fontsize (int, optional):
            Font size of the figure title. Defaults to 20.
        title_y (float, optional):
            Vertical offset for the title in figure coordinates. Defaults to 1.4.
        contribution_threshold (float, optional):
            Minimum absolute SHAP value required for a feature to appear in the
            plot. Defaults to 0.0 (show all).

    Returns:
        plt.Figure:
            The matplotlib figure containing the SHAP force plot.
    """

    # Generate readable feature names
    feature_names = [s.replace("_", " ").title() for s in X.columns]

    # Extract SHAP components for the selected sample
    values = np.round(shap_values[idx].values, 2)
    base_value = np.round(shap_values[idx].base_values, 2)
    feature_values = np.round(X.iloc[idx], 2)

    # Create SHAP force plot (Matplotlib backend)
    fig = shap.plots.force(
        base_value = base_value,
        shap_values = values,
        features = feature_values,
        feature_names = feature_names,
        text_rotation = text_rotation,
        matplotlib = True,
        contribution_threshold = contribution_threshold,
        show = False,
    )

    # Add a descriptive title
    fig.suptitle(
        f"Explanation for {prediction_label} Prediction ({model_name}) [Force Plot]",
        y = title_y,
        fontsize = title_fontsize
    )
    
    return fig