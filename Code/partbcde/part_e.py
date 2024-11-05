# importing the libraries
import numpy as np
from Classes_and_functions import *
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

# Loading breast cancer data
data = load_breast_cancer()
X = data.data
y = data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initiate instance of the Logistic regression class
LR = Logisticregr(X_train, y_train, eta=0.01, lmbd=0.0, iterations=100)

# Calculate training accuracy before hyperparameter tuning
LR.SGD(np.zeros(X_train.shape[1]))
accuracy_train_initial = LR.accuracy(X_train, y_train)
print(f'Initial Training Accuracy: {accuracy_train_initial}')

# Define your hyperparameter ranges
learning_rate_range = np.logspace(-3, -1, 10)
lmb_range = np.logspace(-5, -1, 10)
results_sci = []
results_LR = []

# Track training accuracies for the own logistic regression model
results_LR_train = []

for learning_rate, lmb in product(learning_rate_range, lmb_range):
    # Scikit Learn reference model
    clf = SGDClassifier(loss='log_loss', alpha = lmb, learning_rate='constant', eta0=learning_rate, max_iter=100, random_state=1, penalty='l2')
    clf.fit(X_train, y_train)
    predictsci = clf.predict(X_test)
    accuracy_sci_test = accuracy_score(y_test, predictsci)
    accuracy_sci_train = accuracy_score(y_train, clf.predict(X_train))

    results_sci.append((learning_rate, lmb, accuracy_sci_test, accuracy_sci_train))

    # Own Logistic Regression model
    LR_classifier = Logisticregr(X_train, y_train, eta=learning_rate, lmbd=lmb, iterations=100, nbatches=10)
    LR_classifier.SGD(np.zeros(X_train.shape[1]))

    # Calculate predictions and accuracy for the test set
    accuracy_LR_test = LR_classifier.accuracy(X_test, y_test)
    accuracy_LR_train = LR_classifier.accuracy(X_train, y_train)
    results_LR.append((learning_rate, lmb, accuracy_LR_test))
    results_LR_train.append((learning_rate, lmb, accuracy_LR_train))

# Convert results to structured format
results_LR = np.array(results_LR, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8')])
results_LR_train = np.array(results_LR_train, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8')])
results_sci = np.array(results_sci, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy_test', 'f8'), ('accuracy_train', 'f8')])

# Reshape accuracy values for training and test
accuracy_values_sci_test = results_sci['accuracy_test'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_sci_train = results_sci['accuracy_train'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_LR_test = results_LR['accuracy'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_LR_train = results_LR_train['accuracy'].reshape(len(learning_rate_range), len(lmb_range))

# Combined plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Scikit-learn test accuracy heatmap
sns.heatmap(accuracy_values_sci_test.T, annot=True, fmt=".3f",
            xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_range],
            yticklabels=[f"{np.log10(lmb):.2f}" for lmb in lmb_range],
            ax=axes[0, 1], cmap="YlGnBu")
axes[0, 1].set_title('Test Accuracy Heatmap for Scikit-learn Logistic Regression (SGDClassifier)')
axes[0, 1].set_xlabel('Log Learning Rate')
axes[0, 1].set_ylabel('Log Lambda (L2 Regularization)')

# Scikit-learn train accuracy heatmap
sns.heatmap(accuracy_values_sci_train.T, annot=True, fmt=".3f",
            xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_range],
            yticklabels=[f"{np.log10(lmb):.2f}" for lmb in lmb_range],
            ax=axes[0, 0], cmap="YlGnBu")
axes[0, 0].set_title('Training Accuracy Heatmap for Scikit-learn Logistic Regression (SGDClassifier)')
axes[0, 0].set_xlabel('Log Learning Rate')
axes[0, 0].set_ylabel('Log Lambda (L2 Regularization)')

# Own Logistic Regression test accuracy heatmap
sns.heatmap(accuracy_values_LR_test.T, annot=True, fmt=".3f",
            xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_range],
            yticklabels=[f"{np.log10(lmb):.2f}" for lmb in lmb_range],
            ax=axes[1, 1], cmap="YlGnBu")
axes[1, 1].set_title('Test Accuracy Heatmap for Custom Logistic Regression')
axes[1, 1].set_xlabel('Log Learning Rate')
axes[1, 1].set_ylabel('Log Lambda (L2 Regularization)')

# Own Logistic Regression train accuracy heatmap
sns.heatmap(accuracy_values_LR_train.T, annot=True, fmt=".3f",
            xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_range],
            yticklabels=[f"{np.log10(lmb):.2f}" for lmb in lmb_range],
            ax=axes[1, 0], cmap="YlGnBu")
axes[1, 0].set_title('Training Accuracy Heatmap for Custom Logistic Regression')
axes[1, 0].set_xlabel('Log Learning Rate')
axes[1, 0].set_ylabel('Log Lambda (L2 Regularization)')

plt.tight_layout()
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Combined_Accuracy_Heatmaps.png')
plt.show()
