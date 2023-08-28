# TODO After we have a working solution, we can compare all classifiers from
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# From generator.py
max_agents = 10
max_test_nodes = 5

num_input_nodes = max_agents * max_test_nodes + max_test_nodes + 1

data = np.genfromtxt("./data/agents.data", delimiter=',', dtype=int)

# TODO: Optimize splitting
X = np.array([row[:len(row) - 1] for row in data])
y = np.array([[row[len(row) - 1]] for row in data])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

# Logistic regression
# reg_log = LogisticRegression()
# reg_log.fit(X_train, y_train)
# y_pred = reg_log.predict(X_validation)

# print(metrics.classification_report(y_validation, y_pred))

# Random forest decision tree
reg_rf = RandomForestClassifier()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_validation)

print(metrics.classification_report(y_validation, y_pred))


