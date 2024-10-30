# imporing the libraries
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

# Loading breast cancer data
data = load_breast_cancer()
X = data.data
y = data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initate instance of the Logistic regression class

LR = Logisticregr(X_train, y_train, eta=0.01, lmbd=0.0, iterations=100)


LR.SGD(np.zeros(X_train.shape[1]))
predictions = LR.predict(X_test)
print(f'Accuracy of the model: {LR.accuracy(X_test, y_test)}')


# Define your hyperparameter ranges
learning_rate_range = np.logspace(-3, -1, 10)
lmb_range = np.logspace(-5, -1, 10)
results_sci = []
results_LR = []

for learning_rate, lmb in product(learning_rate_range, lmb_range):
    # Scikit Learn reference model: Ref. https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
    clf = LogisticRegression(C=1.0/lmb, solver='lbfgs', max_iter=100, random_state=1, penalty='l2')
    clf.fit(X_train, y_train)
    predictsci = clf.predict(X_test)
    accuracy_sci = accuracy_score(y_test, predictsci)

    results_sci.append((learning_rate, lmb, accuracy_sci))

    # Own Logistic Regression model
    LR_classifier = Logisticregr(X_train, y_train, eta=learning_rate, lmbd=lmb, iterations=100)


    LR_classifier.SGD(np.zeros(X_train.shape[1]))
    predictions = LR_classifier.predict(X_test)
    print(f'Accuracy of the model: {LR_classifier.accuracy(X_test, y_test)}')
    

    predictions = LR_classifier.predict(X_test)
   
    accuracy_LR = LR_classifier.accuracy(X_test, y_test)
   

    results_LR.append((learning_rate, lmb, accuracy_LR))

# Convert results to a structured format
results_LR = np.array(results_LR, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8')])
results_sci = np.array(results_sci, dtype=[('learning_rate', 'f8'), ('lambda', 'f8'), ('accuracy', 'f8')])

# Debugging prints
print("Shape of results_sci:", results_sci.shape)
print("Learning rate range:", learning_rate_range)
print("Lambda range:", lmb_range)

accuracy_values_sci = results_sci['accuracy'].reshape(len(learning_rate_range), len(lmb_range))
accuracy_values_LR = results_LR['accuracy'].reshape(len(learning_rate_range), len(lmb_range))
# Plotting

plt.figure(figsize=(10, 10))
sns.heatmap(accuracy_values_sci, annot=True, fmt=".3f", 
            xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_range],
            yticklabels=[f"{np.log10(lmb):.2f}" for lmb in lmb_range],
            cmap="YlGnBu")  
plt.title('Accuracy Heatmap (Scikit-learn Logisitic Regression)')
plt.xlabel('Log Learning Rate')
plt.ylabel('Log Lambda (L2 Regularization)')
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_Scikit_Logistic_regression.png')
plt.show()


plt.figure(figsize=(10, 10))
sns.heatmap(accuracy_values_LR, annot=True, fmt=".3f", 
            xticklabels=[f"{np.log10(lr):.2f}" for lr in learning_rate_range],
            yticklabels=[f"{np.log10(lmb):.2f}" for lmb in lmb_range],
            cmap="YlGnBu")  
plt.title('Accuracy Heatmap (Own Logistic Regression)')
plt.xlabel('Log Learning Rate')
plt.ylabel('Log Lambda (L2 Regularization)')
plt.savefig(r'G:\My Drive\UIO\Subjects\FYS-STK4155\Oppgaver\Projects\Project 2\Figures\Heatmap_OwnLogistic.png')
plt.show()