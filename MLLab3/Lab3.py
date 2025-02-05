import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from scipy.spatial.distance import minkowski

# Load dataset
df = pd.read_excel("C:\\Users\\mrhim\\OneDrive\\Desktop\\ML\\ML_Assignment4\\ML\\10-Java_AST_in_.xlsx")

# Compute median of Final_Marks and create binary target
median_final_marks = df["Final_Marks"].median()
df["Trend"] = (df["Final_Marks"] > median_final_marks).astype(int)

# Select features dynamically based on dataset structure
feature_columns = [col for col in df.columns if col.startswith("ast_")]
X = df[feature_columns].fillna(df[feature_columns].mean())
y = df["Trend"]

# A1: Intraclass spread and interclass distances
class1_vectors, class2_vectors = X[y == 1], X[y == 0]
centroid1, centroid2 = np.mean(class1_vectors, axis=0), np.mean(class2_vectors, axis=0)
spread1, spread2 = np.std(class1_vectors, axis=0), np.std(class2_vectors, axis=0)
distance_between_centroids = np.linalg.norm(centroid1 - centroid2)
print("Class 1 Centroid:", centroid1)
print("Class 2 Centroid:", centroid2)
print("Class 1 Spread:", spread1)
print("Class 2 Spread:", spread2)
print("Distance between centroids:", distance_between_centroids)

# A2: Histogram for one feature
feature_index = 'ast_0'  # Change if needed
feature_data = X[feature_index]
plt.hist(feature_data, bins=10, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Feature Values')
plt.ylabel('Frequency')
plt.title(f'Histogram of {feature_index}')
plt.show()
print("Mean of Feature:", np.mean(feature_data))
print("Variance of Feature:", np.var(feature_data))