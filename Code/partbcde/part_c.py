import numpy as np
from Classes_and_functions import *
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import logspace
from itertools import product
from sklearn.metrics import r2_score

# Data generation
np.random.seed(2014)
n = 200  # Number of data points
x = np.random.rand(n, 1)  # Input
y = 2 + 3 * x + 4 * x**2 + 0.1 * np.random.randn(n, 1)

p = 2  # Polynomial degree
X = Design(x, p)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define your hyperparameter ranges
learning_rate = np.logspace(-3, -1, 10)
lmb = np.logspace(-5, -1, 10)

# Create a grid of parameters
grid = product(lmb, learning_rate)

# Initialize a list to store results for ReLU
results = []

# ReLU activation
for lmb_value, lr in grid:
    nn_relu = NetworkClass(
        network_input_size=X_train.shape[1],
        layer_output_sizes=[5, 1],
        activation_funcs=[ReLU, linear],
        activation_ders=[ReLU_der, linear_derivative],
        cost_fun=mean_squared_error,
        cost_der=mean_squared_error_derivative
    )

    for epoch in range(100):
        layer_grads = nn_relu.compute_gradient(X_train, y_train)
        nn_relu.update_weights(layer_grads, lr, lmb_value)

    predictions = nn_relu.predict(X_test)
    mse_nn_relu = mean_squared_error(predictions, y_test)
    R2_nn_relu = r2_score(y_test, predictions)
    results.append((lmb_value, lr, mse_nn_relu, R2_nn_relu))

# Convert results to a structured format
results_array = np.array(results, dtype=[('lambda', 'f8'), ('learning_rate', 'f8'), ('mse', 'f8'), ('R2_nn', 'f8')])

# Reshape the results for heatmap
mse_values = results_array['mse'].reshape(len(lmb), len(learning_rate))
R2_values = results_array['R2_nn'].reshape(len(lmb), len(learning_rate))

mse_minimum = results_array[np.argmin(results_array['mse'])]
R2_maximum = results_array[np.argmax(results_array['R2_nn'])]

# Create two heatmaps side by side one for mse and one for R2
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Heatmap for MSE
sns.heatmap(mse_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[0])
axes[0].set_title('Mean Squared Error Heatmap (Custom NN with ReLU)')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Lambda (L2 Regularization)')

# Heatmap for R2
sns.heatmap(R2_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[1])
axes[1].set_title('R2 Score Heatmap (Custom NN with ReLU)')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Lambda (L2 Regularization)')

plt.tight_layout()
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_ReLU.png')
plt.show()

print("ReLU activation. Best lambda and learning rate with respect to lowest MSE: ", mse_minimum['lambda'], mse_minimum['learning_rate'])
print("Lowest MSE: ", mse_minimum['mse'])
print("ReLU activation. Best lambda and learning rate with respect to highest R2: ", R2_maximum['lambda'], R2_maximum['learning_rate'])
print("Highest R2: ", R2_maximum['R2_nn'])


# Leaky ReLU activation
# Define the grid of hyperparameters

learning_rate = np.logspace(-3, -1, 10)
lmb = np.logspace(-5, -1, 10)

# Create a grid of parameters
grid = product(lmb, learning_rate)

results_leaky = []
# ReLU activation
for lmb_value, lr in grid:
    nn_leaky = NetworkClass(
        network_input_size=X_train.shape[1],
        layer_output_sizes=[5, 1],
        activation_funcs=[leaky_ReLU, linear],
        activation_ders=[der_leaky_ReLU, linear_derivative],
        cost_fun=mean_squared_error,
        cost_der=mean_squared_error_derivative
    )

    for epoch in range(100):
        layer_grads = nn_leaky.compute_gradient(X_train, y_train)
        nn_leaky.update_weights(layer_grads, lr, lmb_value)

    predictions = nn_leaky.predict(X_test)
    mse_nn_leaky = mean_squared_error(predictions, y_test)
    R2_nn_leaky = r2_score(y_test, predictions)
    results_leaky.append((lmb_value, lr, mse_nn_leaky, R2_nn_leaky))

# Convert results to a structured format
results_array_leaky = np.array(results_leaky, dtype=[('lambda', 'f8'), ('learning_rate', 'f8'), ('mse', 'f8'), ('R2_nn', 'f8')])

# Reshape the results for heatmap
mse_values = results_array_leaky['mse'].reshape(len(lmb), len(learning_rate))
R2_values = results_array_leaky['R2_nn'].reshape(len(lmb), len(learning_rate))

mse_minimum = results_array[np.argmin(results_array['mse'])]
R2_maximum = results_array[np.argmax(results_array['R2_nn'])]

# Create two heatmaps side by side one for mse and one for R2
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Heatmap for MSE
sns.heatmap(mse_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[0])
axes[0].set_title('Mean Squared Error Heatmap (Custom NN with leaky ReLU)')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Lambda (L2 Regularization)')

# Heatmap for R2
sns.heatmap(R2_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[1])
axes[1].set_title('R2 Score Heatmap (Custom NN with leaky ReLU)')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Lambda (L2 Regularization)')

plt.tight_layout()
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_Leaky_ReLU.png')
plt.show()

print("Leaky ReLU activation. Best lambda and learning rate with respect to lowest MSE: ", mse_minimum['lambda'], mse_minimum['learning_rate'])
print("Lowest MSE: ", mse_minimum['mse'])
print("Leaky ReLU activation. Best lambda and learning rate with respect to highest R2: ", R2_maximum['lambda'], R2_maximum['learning_rate'])
print("Highest R2: ", R2_maximum['R2_nn'])


