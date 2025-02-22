import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\mrhim\OneDrive\Desktop\ML\ML_AS4\10-Java_AST_in_.xlsx"
df = pd.read_excel(file_path, sheet_name="in")

# features and target
X = df.drop(columns=["Final_Marks", "error_count"])  # Remove target variables
y = df["Final_Marks"]  # Target for regression

# training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A1 & A3: Train Linear Regression Model
reg = LinearRegression().fit(X_train, y_train)
y_train_pred, y_test_pred = reg.predict(X_train), reg.predict(X_test)

# A2: Compute performance metrics
def compute_metrics(y_true, y_pred, dataset="Train"):
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
    print(f"{dataset} Metrics: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAPE={metrics['MAPE']:.4f}, R2={metrics['R2']:.4f}")

compute_metrics(y_train, y_train_pred, "Train")
compute_metrics(y_test, y_test_pred, "Test")

# A4: K-Means Clustering (ignoring target variable)
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_train)

# A5: Compute clustering scores
scores = {
    "Silhouette Score": silhouette_score(X_train, kmeans.labels_),
    "CH Score": calinski_harabasz_score(X_train, kmeans.labels_),
    "DB Index": davies_bouldin_score(X_train, kmeans.labels_)
}
print(f"Silhouette Score: {scores['Silhouette Score']:.4f}, CH Score: {scores['CH Score']:.4f}, DB Index: {scores['DB Index']:.4f}")

# A6 & A7: Evaluate K-Means for multiple k values
k_values = range(2, 20)
results = {
    "sil_scores": [],
    "ch_scores": [],
    "db_scores": [],
    "distortions": []
}

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_train)
    results["distortions"].append(kmeans.inertia_)
    results["sil_scores"].append(silhouette_score(X_train, kmeans.labels_))
    results["ch_scores"].append(calinski_harabasz_score(X_train, kmeans.labels_))
    results["db_scores"].append(davies_bouldin_score(X_train, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(k_values, results["distortions"], marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Plot for Optimal k')
plt.show()

# Plot clustering scores
plt.figure(figsize=(12, 5))
plt.plot(k_values, results["sil_scores"], marker='o', label='Silhouette Score')
plt.plot(k_values, results["ch_scores"], marker='s', label='Calinski-Harabasz Score')
plt.plot(k_values, results["db_scores"], marker='^', label='Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.legend()
plt.title('Clustering Evaluation Metrics')
plt.show()