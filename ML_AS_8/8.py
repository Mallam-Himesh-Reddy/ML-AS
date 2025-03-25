import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Define AND and XOR gate datasets
def get_logic_gate_data(gate_type="AND"):
    if gate_type == "AND":
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 0, 0, 1])
    elif gate_type == "XOR":
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 1, 1, 0])
    return X, y

# Activation Functions
def step_function(x):
    return 1 if x >= 0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

# Perceptron Training
def perceptron_train(X, y, activation="step", learning_rate=0.05, epochs=100):
    num_inputs = X.shape[1]
    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    errors = []
    
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(X)):
            z = np.dot(X[i], weights) + bias
            if activation == "step":
                y_pred = step_function(z)
            elif activation == "sigmoid":
                y_pred = 1 if sigmoid(z) >= 0.5 else 0
            elif activation == "relu":
                y_pred = 1 if relu(z) >= 0.5 else 0
            else:
                raise ValueError("Unknown activation function")
            
            error = y[i] - y_pred
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
            total_error += abs(error)
        
        errors.append(total_error)
        if total_error == 0:
            break
    
    return weights, bias, errors

# Train Perceptron for AND Gate
X_and, y_and = get_logic_gate_data("AND")
weights_and, bias_and, errors_and = perceptron_train(X_and, y_and, activation="step")

# Train Perceptron for XOR Gate
X_xor, y_xor = get_logic_gate_data("XOR")
weights_xor, bias_xor, errors_xor = perceptron_train(X_xor, y_xor, activation="step")

# Plot Error Reduction
plt.plot(errors_and, label="AND Gate")
plt.plot(errors_xor, label="XOR Gate")
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.title("Perceptron Training Error Reduction")
plt.legend()
plt.show()

# Print Results
print(f"AND Gate Perceptron - Weights: {weights_and}, Bias: {bias_and}")
print(f"XOR Gate Perceptron - Weights: {weights_xor}, Bias: {bias_xor}")

# MLP Training for AND/XOR Gates
mlp_and = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=1000, solver='adam', random_state=42)
mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=1000, solver='adam', random_state=42)

mlp_and.fit(X_and, y_and)
mlp_xor.fit(X_xor, y_xor)

y_pred_and = mlp_and.predict(X_and)
y_pred_xor = mlp_xor.predict(X_xor)

# Accuracy of MLP
acc_and = accuracy_score(y_and, y_pred_and)
acc_xor = accuracy_score(y_xor, y_pred_xor)

print(f"MLP Accuracy for AND Gate: {acc_and:.4f}")
print(f"MLP Accuracy for XOR Gate: {acc_xor:.4f}")
