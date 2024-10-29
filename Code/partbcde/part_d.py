import numpy as np
from Classes_and_functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

# Loading data
data = load_breast_cancer()
X = data.data
y = data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X.shape)
# Define your hyperparameter ranges
learning_rate_range = np.logspace(-3, -1, 10)
lmb_range = np.logspace(-5, -1, 10)
results_sci = []
results_nn = []

for learning_rate, lmb in product(learning_rate_range, lmb_range):
    # Scikit Learn reference model: Ref. https://scikit-learn.org/1.5/modules/neural_networks_supervised.html
    clf = MLPClassifier(learning_rate_init=learning_rate, solver='adam', alpha=lmb, activation='logistic',
                        hidden_layer_sizes=(5), random_state=1, max_iter=10)
    clf.fit(X_train, y_train)
    predictsci = clf.predict(X_test)
    accuracy_sci = accuracy_score(y_test, predictsci)

    results_sci.append((learning_rate, lmb, accuracy_sci))

    # Neural Network model
    nn_classifier = NetworkClass(network_input_size=30,
                       layer_output_sizes=[5, 1],
                       activation_funcs=[sigmoid, sigmoid],
                       activation_ders=[sigmoid_derivative, sigmoid_derivative],
                       cost_fun=CostCrossEntropy,
                       cost_der=CostCrossEntropyDer)
    

    inputs = X_train
    targets = y_train.reshape(-1, 1)
    print("target shape", targets.shape)  # Ensure targets are in the correct shape
    for epoch in range(1):
        layer_grads = nn_classifier.compute_gradient(inputs, targets)
        nn_classifier.update_weights(layer_grads, learning_rate=0.01, lmbd=0.01)

    predictions = nn_classifier.predict(X_test)
    predictions = (predictions > 0.5).astype(int) # Convert predictions to binary
    accuracy_nn = accuracy_score(predictions.flatten(),y_test)
    print(f'Accuracy of the model: {accuracy_score(predictions.flatten(), y_test)}')

    results_nn.append((learning_rate, lmb, accuracy_nn))

# Convert results to a structured format
results_nn = np.array(results_nn, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8')])
results_sci = np.array(results_sci, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8')])

# Debugging prints
print("Shape of results_sci:", results_sci.shape)
print("Learning rate range:", learning_rate_range)
print("Lambda range:", lmb_range)

accuracy_values_sci = results_sci['accuracy'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_nn = results_nn['accuracy'].reshape(len(learning_rate_range), len(lmb_range))
# Plotting

plt.figure(figsize=(10, 10))
sns.heatmap(accuracy_values_sci, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.2e}" for lr in learning_rate_range],
            yticklabels=[f"{lmb:.2e}" for lmb in lmb_range],
            cmap="YlGnBu")  
plt.title('Accuracy Heatmap (Scikit-learn)')
plt.xlabel('Learning Rate')
plt.ylabel('Lambda (L2 Regularization)')
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_Scikit_classification.png')
plt.show()


plt.figure(figsize=(10, 10))
sns.heatmap(accuracy_values_nn, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.2e}" for lr in learning_rate_range],
            yticklabels=[f"{lmb:.2e}" for lmb in lmb_range],
            cmap="YlGnBu")  
plt.title('Accuracy Heatmap (NN)')
plt.xlabel('Learning Rate')
plt.ylabel('Lambda (L2 Regularization)')
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_NN_classification.png')
plt.show()


