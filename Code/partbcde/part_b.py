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

# OLS reference:
beta_ols = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
ytilde = X_train @ beta_ols
ytilde_test = X_test @ beta_ols
mse_ols = mean_squared_error(y_train, ytilde)
R2_ols = r2_score(y_train, ytilde)
print(f'Mean squared error OLS train: {mse_ols}')
print(f'R2 score OLS train: {R2_ols}')
mse_ols_t = mean_squared_error(y_test, ytilde_test)
R2_ols_t = r2_score(y_test, ytilde_test)
print(f'Mean squared error OLS train: {mse_ols_t}')
print(f'R2 score OLS train: {R2_ols_t}')

# Define your hyperparameter ranges
learning_rate = np.logspace(-3, -1, 10)
lmb = np.logspace(-5, -1, 10)

# Create a grid of parameters
grid = product(lmb, learning_rate)

# Initialize lists to store results
results_sci = []

# Loop through all combinations of lmb and learning_rate
for lmb_value, lr in grid:
    # Reset the neural network for each combination
    scikitMLP = MLPRegressor(alpha=lmb_value, activation='identity', hidden_layer_sizes=[5],
                             learning_rate_init=lr, random_state=1, max_iter=1000)
    
    # Fit the model
    scikitMLP.fit(X_train, y_train.ravel())
    
    # Make predictions
    predictsci = scikitMLP.predict(X_test).reshape(-1, 1)

    # Calculate performance metrics
    mse_sci = mean_squared_error(y_test, predictsci)
    R2_sci = scikitMLP.score(X_test, y_test)

    # Store results
    results_sci.append((lmb_value, lr, mse_sci, R2_sci))

# Convert results to a structured format
results_sci_array = np.array(results_sci, dtype=[('lambda', 'f8'), ('learning_rate', 'f8'), 
                                                  ('mse_sci', 'f8'), ('R2_sci', 'f8')])

# Reshape the results for plotting
mse_sci_values = results_sci_array['mse_sci'].reshape(len(lmb), len(learning_rate))
R2_sci_values = results_sci_array['R2_sci'].reshape(len(lmb), len(learning_rate))

# Create two heatmaps side by side one for mse and one for R2
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Heatmap for MSE
sns.heatmap(mse_sci_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[0])
axes[0].set_title('Mean Squared Error Heatmap (Scikit-learn)')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Lambda (L2 Regularization)')

# Heatmap for R2
sns.heatmap(R2_sci_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[1])
axes[1].set_title('R2 Score Heatmap (Scikit-learn)')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Lambda (L2 Regularization)')

plt.tight_layout()
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_MLPRegressor.png')
plt.show()

# Own Neural Network training example
inputs = X_train
targets = y_train

# Define your hyperparameter ranges
learning_rate = np.logspace(-3, -1, 10)
lmb = np.logspace(-5, -1, 10)

# Create a grid of parameters
grid = product(lmb, learning_rate)

# Initialize a list to store results
results = []

# Loop through all combinations of lmb and learning_rate
for lmb_value, lr in grid:
    # Reset the neural network for each combination
    nn_regr = NetworkClass(
        network_input_size= X_train.shape[1],
        layer_output_sizes=[50, 1],
        activation_funcs=[sigmoid, linear],
        activation_ders=[sigmoid_derivative, linear_derivative],
        cost_fun=mean_squared_error,
        cost_der=mean_squared_error_derivative
    )

    # Training loop for each combination of hyperparameters
    for epoch in range(100):  # Increased number of epochs
        layer_grads = nn_regr.compute_gradient(inputs, targets)
        nn_regr.update_weights(layer_grads, lr, lmb_value)

    # Prediction example
    predictions = nn_regr.predict(X_test)
    
    # Calculate performance metrics
    mse_nn = mean_squared_error(predictions, y_test)
    R2_nn = r2_score(y_test, predictions)
    results.append((lmb_value, lr, mse_nn, R2_nn))

# Convert results to a structured format
results_array = np.array(results, dtype=[('lambda', 'f8'), ('learning_rate', 'f8'), ('mse', 'f8'), ('R2_nn', 'f8')])

# Reshape the results for heatmap
mse_values = results_array['mse'].reshape(len(lmb), len(learning_rate))
mse_minimum = results_array[np.argmin(results_array['mse'])]

# Reshape R2 values for heatmap
R2_values = results_array['R2_nn'].reshape(len(lmb), len(learning_rate))
R2_maximum = results_array[np.argmax(results_array['R2_nn'])]

# Create two heatmaps side by side one for mse and one for R2
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# Heatmap for MSE
sns.heatmap(mse_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[0])
axes[0].set_title('Mean Squared Error Heatmap (Custom NN with Sigmoid)')
axes[0].set_xlabel('Learning Rate')
axes[0].set_ylabel('Lambda (L2 Regularization)')

# Heatmap for R2
sns.heatmap(R2_values, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=[f"{lr:.4f}" for lr in learning_rate], 
            yticklabels=[f"{lmb_val:.4f}" for lmb_val in lmb], ax=axes[1])
axes[1].set_title('R2 Score Heatmap (Custom NN with Sigmoid)')
axes[1].set_xlabel('Learning Rate')
axes[1].set_ylabel('Lambda (L2 Regularization)')

plt.tight_layout()
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_sigmoid.png')
plt.show()

print("Sigmoid activation. Best lambda and learning rate with respect to lowest MSE: ", mse_minimum['lambda'], mse_minimum['learning_rate'])
print("Lowest MSE: ", mse_minimum['mse'])
print("Sigmoid activation. Best lambda and learning rate with respect to highest R2: ", R2_maximum['lambda'], R2_maximum['learning_rate'])
print("Highest R2: ", R2_maximum['R2_nn'])

