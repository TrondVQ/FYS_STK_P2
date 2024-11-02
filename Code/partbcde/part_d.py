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

# Define your hyperparameter ranges
learning_rate_range = np.logspace(-3, -1, 10)
lmb_range = np.logspace(-5, -1, 10)

results_sci = []
results_nn = []

# Train Scikit-learn model
for learning_rate, lmb in product(learning_rate_range, lmb_range):
    clf = MLPClassifier(learning_rate_init=learning_rate, solver='adam', alpha=lmb, activation='logistic',
                        hidden_layer_sizes=(5), random_state=1, max_iter=100)
    clf.fit(X_train, y_train)
    
    # Accuracy on test data
    predictsci = clf.predict(X_test)
    accuracy_sci = accuracy_score(y_test, predictsci)
    
    # Accuracy on train data
    predictsci_train = clf.predict(X_train)
    accuracy_sci_train = accuracy_score(y_train, predictsci_train)

    results_sci.append((learning_rate, lmb, accuracy_sci, accuracy_sci_train))

    # Neural Network model
    nn_classifier = NetworkClass(network_input_size=30,
                                  layer_output_sizes=[5, 1],
                                  activation_funcs=[sigmoid, sigmoid],
                                  activation_ders=[sigmoid_derivative, sigmoid_derivative],
                                  cost_fun=CostCrossEntropy,
                                  cost_der=CostCrossEntropyDer)

    inputs = X_train
    targets = y_train.reshape(-1, 1)
    
    nn_classifier.train(inputs, targets, epochs=100, batch_size=16, learning_rate=learning_rate, lmbd=lmb)   

    predictions = nn_classifier.predict(X_test)
    predictions_train = nn_classifier.predict(X_train)
    
    predictions = (predictions > 0.5).astype(int)  # Convert predictions to binary
    predictions_train = (predictions_train > 0.5).astype(int)  # Convert train predictions to binary
    
    accuracy_nn = accuracy_score(predictions.flatten(), y_test)
    accuracy_nn_train = accuracy_score(predictions_train.flatten(), y_train)

    results_nn.append((learning_rate, lmb, accuracy_nn, accuracy_nn_train))

# Convert results to structured format
results_nn = np.array(results_nn, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8'), ('accuracy_train', 'f8')])
results_sci = np.array(results_sci, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8'), ('accuracy_train', 'f8')])

# Reshape the results for heatmaps
accuracy_values_sci_train = results_sci['accuracy_train'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_sci_test = results_sci['accuracy'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_nn_train = results_nn['accuracy_train'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_nn_test = results_nn['accuracy'].reshape(len(learning_rate_range), len(lmb_range))

# Plotting heatmaps
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Scikit-learn Train Heatmap
sns.heatmap(accuracy_values_sci_train, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.2e}" for lr in learning_rate_range],
            yticklabels=[f"{lmb:.2e}" for lmb in lmb_range],
            cmap="YlGnBu", ax=axes[0, 0])  
axes[0, 0].set_title('Train Accuracy Heatmap (Scikit-learn)')
axes[0, 0].set_xlabel('Learning Rate')
axes[0, 0].set_ylabel('Lambda (L2 Regularization)')

# Scikit-learn Test Heatmap
sns.heatmap(accuracy_values_sci_test, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.2e}" for lr in learning_rate_range],
            yticklabels=[f"{lmb:.2e}" for lmb in lmb_range],
            cmap="YlGnBu", ax=axes[0, 1])  
axes[0, 1].set_title('Test Accuracy Heatmap (Scikit-learn)')
axes[0, 1].set_xlabel('Learning Rate')
axes[0, 1].set_ylabel('Lambda (L2 Regularization)')

# Custom NN Train Heatmap
sns.heatmap(accuracy_values_nn_train, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.2e}" for lr in learning_rate_range],
            yticklabels=[f"{lmb:.2e}" for lmb in lmb_range],
            cmap="YlGnBu", ax=axes[1, 0])  
axes[1, 0].set_title('Train Accuracy Heatmap (Custom NN)')
axes[1, 0].set_xlabel('Learning Rate')
axes[1, 0].set_ylabel('Lambda (L2 Regularization)')

# Custom NN Test Heatmap
sns.heatmap(accuracy_values_nn_test, annot=True, fmt=".3f", 
            xticklabels=[f"{lr:.2e}" for lr in learning_rate_range],
            yticklabels=[f"{lmb:.2e}" for lmb in lmb_range],
            cmap="YlGnBu", ax=axes[1, 1])  
axes[1, 1].set_title('Test Accuracy Heatmap (Custom NN)')
axes[1, 1].set_xlabel('Learning Rate')
axes[1, 1].set_ylabel('Lambda (L2 Regularization)')

plt.tight_layout()
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_Combined.png')
plt.show()
