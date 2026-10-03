"""Week 2 - Supervised ML model training and evaluation.
Run from the repository root: python src/train_models.py
"""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# THINK: Load the Titanic data from Week 1 and keep only features that are useful for modeling.
df = pd.read_csv(DATA / "titanic.csv")
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X, y = df[features], df['Survived']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_features = ['Pclass','Age','SibSp','Parch','Fare']
categorical_features = ['Sex','Embarked']

def make_preprocessor(scale_numeric: bool):
    num_steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(num_steps), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

models = {
    'Logistic Regression': Pipeline([
        ('preprocessor', make_preprocessor(True)),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', make_preprocessor(False)),
        ('model', DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', make_preprocessor(False)),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('preprocessor', make_preprocessor(True)),
        ('model', KNeighborsClassifier(n_neighbors=5))
    ])
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall': recall_score(y_test, pred, zero_division=0),
        'F1 Score': f1_score(y_test, pred, zero_division=0),
    })
    joblib.dump(model, RESULTS / f"{name.lower().replace(' ','_').replace('-','')}_pipeline.joblib")

pd.DataFrame(rows).to_csv(RESULTS / 'classification_results.csv', index=False)

# THINK: Linear Regression needs a continuous target, so use sklearn's built-in diabetes regression dataset.
diabetes = load_diabetes(as_frame=True)
reg = diabetes.frame
Xr, yr = reg.drop(columns=['target']), reg['target']
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.20, random_state=42)
lin = LinearRegression()
lin.fit(Xr_train, yr_train)
pred = lin.predict(Xr_test)
pd.DataFrame([{
    'Model': 'Linear Regression',
    'MAE': mean_absolute_error(yr_test, pred),
    'MSE': mean_squared_error(yr_test, pred),
    'R2 Score': r2_score(yr_test, pred)
}]).to_csv(RESULTS / 'regression_results.csv', index=False)
joblib.dump(lin, RESULTS / 'linear_regression.joblib')

print('Done. Results written to the results/ directory.')
