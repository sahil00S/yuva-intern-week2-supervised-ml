# YuvaIntern Week 2 - Supervised Machine Learning Model Comparison

## Project Overview
This project implements supervised machine learning algorithms using Python and Scikit-learn. It contains a classification workflow on the Titanic dataset and a regression workflow using Scikit-learn's built-in Diabetes dataset. The project demonstrates data splitting, preprocessing, model training, prediction, metric-based evaluation, and model comparison.

## Algorithms Implemented
### Classification
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)

### Regression
- Linear Regression

## Classification Dataset
The Titanic dataset contains 891 records and 12 original columns. The modelling target is `Survived`. The selected features are `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`. The same dataset was used in the Week 1 preprocessing project.

## Preprocessing
- Numerical missing values: median imputation
- Categorical missing values: most-frequent imputation
- Categorical variables: one-hot encoding
- Scaling: StandardScaler for Logistic Regression and KNN
- Tree models: no numeric scaling required
- Train/test split: 80/20 with `random_state=42` and stratification for classification

## Regression Dataset
Scikit-learn's built-in Diabetes dataset is used for Linear Regression. It contains 442 rows, 10 input features, and a continuous target.

## Evaluation Metrics
### Classification
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Regression
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

## Results
Classification results are stored in `results/classification_results.csv`. Regression results are stored in `results/regression_results.csv`.

### Classification results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.8156 | 0.8000 | 0.6957 | 0.7442 |
| Random Forest | 0.8156 | 0.8103 | 0.6812 | 0.7402 |
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 |

### Regression results

| Model | MAE | MSE | R² |
|---|---:|---:|---:|
| Linear Regression | 42.7941 | 2900.1936 | 0.4526 |

## Repository Structure
```text
yuva-intern-week2-supervised-ml/
├── data/
│   ├── titanic.csv
│   ├── titanic_model_input.csv
│   └── diabetes.csv
├── figures/
│   ├── classification_model_comparison.png
│   ├── random_forest_confusion_matrix.png
│   └── linear_regression_actual_vs_predicted.png
├── notebooks/
│   └── supervised_ml_comparison.ipynb
├── report/
│   └── Week2_Supervised_ML_Report.docx
├── results/
│   ├── classification_results.csv
│   ├── regression_results.csv
│   ├── experiment_summary.json
│   └── model pipeline files
├── src/
│   └── train_models.py
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/train_models.py
```
Or open and run `notebooks/supervised_ml_comparison.ipynb` in Jupyter Notebook or VS Code.

## Reproducibility
A fixed `random_state=42` is used for the train/test splits and model initialisation where applicable. Preprocessing is kept inside Scikit-learn pipelines so transformations are fitted on the training data and then applied to the test data.

## Limitations
The datasets are educational benchmark datasets, so the measured performance should not be interpreted as production performance. Feature selection is intentionally kept beginner-friendly to match the scope of the internship task.
