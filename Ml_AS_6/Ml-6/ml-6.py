import pandas as pd
import numpy as np
from math import log2
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_excel(r"C:\Users\mrhim\OneDrive\Desktop\ML\Ml-6\10-Java_AST_in_.xlsx")

# A1: Calculate Entropy 
def calculate_entropy(column):
    counts = column.value_counts(normalize=True)
    return -np.sum(counts * np.log2(counts))

entropy_value = calculate_entropy(df['ast_0'])
print("A1 - Entropy (ast_0):", entropy_value)

# A2: Calculate Gini Index
def calculate_gini(column):
    counts = column.value_counts(normalize=True)
    return 1 - np.sum(counts**2)

gini_value = calculate_gini(df['ast_0'])
print("A2 - Gini Index (ast_0):", gini_value)

# A3: Information Gain
def information_gain(df, feature_col, target_col):
    total_entropy = calculate_entropy(df[target_col])
    weighted_entropy = df.groupby(feature_col)[target_col].apply(calculate_entropy).mean()
    return total_entropy - weighted_entropy

info_gain = information_gain(df, 'ast_0', 'Final_Marks')
print("A3 - Information Gain for ast_0:", info_gain)

# A4: Equal Width Binning 
df['ast_0_binned'] = pd.cut(df['ast_0'], bins=4)
print("A4 - Binning done: \n", df[['ast_0', 'ast_0_binned']].head())

# A5: Decision Tree Module
features = ['ast_0', 'ast_1']
target = 'Final_Marks'

X = df[features]
y = df[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_model.fit(X_train, y_train)

# A6: Visualize Decision Tree 
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=features, class_names=True, filled=True)
plt.title("Decision Tree")
plt.show()
    
# A7: Decision Boundary Visualization 
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

Z = dt_model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(14, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Set3)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolors='k', cmap='Set1')
plt.xlabel(features[0])
plt.ylabel(features[1])
plt.title("Decision Boundary")
plt.show()