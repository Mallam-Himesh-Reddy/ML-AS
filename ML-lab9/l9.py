import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular

# Load the Excel file
file_path = r"D:\Documents\ML\ML_AS_7\10-Java_AST_in_.xlsx"
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, sheet_name="in")

# Display dataset info
print("Dataset Information:\n")
print(df.info())
print("\nFirst 5 Rows:\n")
print(df.head())

# Define features (X) and target (y)
X = df.drop(columns=['Final_Marks', 'error_count'])  
y = df['Final_Marks']  #

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base regressors
base_models = [
    ('ridge', Ridge()),
    ('dt', DecisionTreeRegressor()),
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('svr', SVR())
]

# Define Stacking Regressor with Ridge as the final estimator
stacking_regressor = StackingRegressor(estimators=base_models, final_estimator=Ridge())

# Train the model
stacking_regressor.fit(X_train, y_train)


train_score = stacking_regressor.score(X_train, y_train)
test_score = stacking_regressor.score(X_test, y_test)

print(f"\nStacking Regressor Train Score: {train_score:.4f}")
print(f"Stacking Regressor Test Score: {test_score:.4f}")

# Create a pipeline with scaling and stacking regressor
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('stacking', stacking_regressor)
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
train_score_pipeline = pipeline.score(X_train, y_train)
test_score_pipeline = pipeline.score(X_test, y_test)

print(f"\nPipeline Train Score: {train_score_pipeline:.4f}")
print(f"Pipeline Test Score: {test_score_pipeline:.4f}")

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    mode="regression"
)

# Explain a single prediction (first instance in the test set)
idx = 0  # Example index
exp = explainer.explain_instance(X_test.iloc[idx].values, pipeline.predict)
lime_explanation = exp.as_list()

# Print LIME explanation
print("\nLIME Explanation (Feature Importance for a Single Prediction):\n")
for feature, importance in lime_explanation:
    print(f"{feature}: {importance:.4f}")

# (Optional) Visualize LIME explanation
exp.show_in_notebook()