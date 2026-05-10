# Machine Learning Explainability Using LIME and SHAP on Heart Disease Data

## Summary

This project focuses on the usage of Local Interpretable Model-agnostic Explanations (LIME) and Shapley Additive Explanations (SHAP) in machine learning for providing human-interpretable explanations for model predictions. Machine learning interpretability / explainability (MLI / MLE) is an important part of the data science process, as it provides a deeper understanding of how models are forming their predictions, even if the internal workings of the model are unknown. MLE increases trust in model predictions and can help identify weaknesses or biases present in machine learning models, allowing for future corrections in model design and data curation.

A heart disease dataset was used which predicted patient death based on a number of heart-health measurements. Four different models were trained to predict patient death based on these measurements, and the results were analyzed using LIME and SHAP to understand which features were being favored by the models for prediction. The explanations generated were compared with domain knowledge to determine the feasibility and trustworthiness of the predictions, allowing for a detailed analysis of the strengths and weaknesses of each model's performance on the data and to contrast the usefulness of each machine learning explainability technique.

## Requirements

The libraries and version of Python used to create this project are listed below. The requirements are also available at [requirements.txt](https://github.com/JoshuaGottlieb/MLI-LIME-SHAP/blob/main/requirements.txt).

```
eli5==0.16.0
lime==0.2.0.1
matplotlib==3.10.6
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
seaborn==0.13.2
shap==0.47.1
xgboost==2.1.4
```

## Repository Structure

```
├── data                                           # Raw and processed train/test datasets
│   ├── heart_failure_clinical_records_dataset.csv
│   ├── heart_failure_test.csv
│   └── heart_failure_train.csv
├── metrics                                        # Classification metrics and predictions for each model
│   ├── model_metrics.csv
│   └── model_predictions.csv
├── models                                         # Pickled and compressed trained models
│   ├── dt.pickle.xz                                   # Decision Tree
│   ├── logreg.pickle.xz                               # Logistic Regression
│   ├── preprocessor.pickle.xz                         # Preprocessing Pipeline
│   ├── rf.pickle.xz                                   # Random Forest
│   └── xgb.pickle.xz                                  # XGBoost
├── src                                            # Project notebooks and source code
│   ├── Explainability.ipynb                           # Notebook containing EDA, training, and analysis
│   ├── modules                                        # Source code with custom functions
│   │   ├── io_utils.py                                    # Functions for serialization and deserialization
│   │   ├── plotting.py                                    # Functions for Matplotlib and Seaborn plotting
│   │   ├── plotting_utils.py                              # Helper functions for plotting
│   │   ├── preprocessing.py                               # Functions for preprocessing raw data and decoding feature matrices
│   │   ├── statistics.py                                  # Functions for EDA statistical techniques
│   │   └── training.py                                    # Functions for training models and gathering predictions
├── README.md
└── requirements.txt
```
