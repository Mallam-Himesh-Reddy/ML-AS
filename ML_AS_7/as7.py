import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint

# Load dataset
file_path = r"D:\Documents\ML\ML_AS_7\10-Java_AST_in_.xlsx"
df = pd.read_excel(file_path, sheet_name="in")

# Preprocess data
df = df.fillna(0).apply(pd.to_numeric, errors='coerce').fillna(0)
X = df.drop(columns=["Final_Marks", "error_count"], errors='ignore')
y = df["Final_Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "SVR": SVR(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "MLPRegressor": MLPRegressor(max_iter=500)
}

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        "Train R²": r2_score(y_train, y_train_pred),
        "Test R²": r2_score(y_test, y_test_pred),
        "Train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred)
    }
    return metrics

# Train and evaluate models
results = {name: evaluate_model(model, X_train, X_test, y_train, y_test) for name, model in models.items()}

# Display results
results_df = pd.DataFrame(results).T
print("\nRegression Model Performance:\n")
print(results_df)

# Hyperparameter tuning for RandomForest
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None] + list(range(5, 30)),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5),
    'max_features': ['auto', 'sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    scoring='r2',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Best model performance
best_rf = random_search.best_estimator_
y_test_pred = best_rf.predict(X_test)
y_train_pred = best_rf.predict(X_train)

print("\nBest RandomForest Parameters:", random_search.best_params_)
print("Best CV Score (R²):", random_search.best_score_)
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))