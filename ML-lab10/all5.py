# Feature Reduction and Explainability Analysis
# This notebook performs:
# - Correlation analysis
# - PCA with 99% and 95% explained variance
# - Sequential Feature Selection
# - Model evaluation
# - LIME and SHAP explanations

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import classification_report, accuracy_score

import shap
import lime
import lime.lime_tabular

# Load dataset
df = pd.read_excel(r"D:\Documents\ML\ML_AS_7\10-Java_AST_in_.xlsx")
print("Initial dataset shape:", df.shape)

# Drop non-numeric columns or handle appropriately
df = df.select_dtypes(include=[np.number])
df = df.dropna()  # Drop rows with missing values
print("Numeric dataset shape after cleaning:", df.shape)

# Split features and target (assume last column is target for this example)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# A1: Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X_scaled, columns=X.columns).corr(), cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# A2: PCA with 99% Variance
pca_99 = PCA(n_components=0.99)
X_pca_99 = pca_99.fit_transform(X_scaled)
print("Shape after PCA 99%:", X_pca_99.shape)

X_train, X_test, y_train, y_test = train_test_split(X_pca_99, y, test_size=0.2, random_state=42)
model_99 = RandomForestClassifier()
model_99.fit(X_train, y_train)
y_pred_99 = model_99.predict(X_test)
print("\nA2 - PCA 99% Accuracy:", accuracy_score(y_test, y_pred_99))
print(classification_report(y_test, y_pred_99))

# A3: PCA with 95% Variance
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
print("Shape after PCA 95%:", X_pca_95.shape)

X_train, X_test, y_train, y_test = train_test_split(X_pca_95, y, test_size=0.2, random_state=42)
model_95 = RandomForestClassifier()
model_95.fit(X_train, y_train)
y_pred_95 = model_95.predict(X_test)
print("\nA3 - PCA 95% Accuracy:", accuracy_score(y_test, y_pred_95))
print(classification_report(y_test, y_pred_95))

# A4: Sequential Feature Selection
base_model = LogisticRegression(max_iter=1000)
sfs = SequentialFeatureSelector(base_model, n_features_to_select='auto', direction='forward')
sfs.fit(X_scaled, y)
X_selected = sfs.transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model_sfs = base_model
model_sfs.fit(X_train, y_train)
y_pred_sfs = model_sfs.predict(X_test)
print("\nA4 - Sequential Feature Selection Accuracy:", accuracy_score(y_test, y_pred_sfs))
print(classification_report(y_test, y_pred_sfs))

# A5: LIME & SHAP Explainability

# LIME
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_scaled, 
    feature_names=X.columns.tolist(), 
    class_names=np.unique(y).astype(str), 
    discretize_continuous=True
)
i = 1
exp = explainer_lime.explain_instance(X_scaled[i], model_99.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True)

# SHAP
explainer_shap = shap.TreeExplainer(model_99)
shap_values = explainer_shap.shap_values(X_scaled[:100])
shap.summary_plot(shap_values,X.iloc[:100])