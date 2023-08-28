# TODO After we have a working solution, we can compare all classifiers from
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from imblearn.over_sampling import SMOTE

from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



# From generator.py
max_agents = 10
max_test_nodes = 5

num_input_nodes = max_agents * max_test_nodes + max_test_nodes + 1

data = np.genfromtxt("./data/agents.data", delimiter=',', dtype=int)

# TODO: Optimize splitting
X = np.array([row[:len(row) - 1] for row in data])
y = np.array([row[len(row) - 1] for row in data])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

print("Destribution before...")
counter = Counter(y_train)
print(counter)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

print("Destribution after...")
counter = Counter(y_train)
print(counter)

# Logistic regression
reg_log = LogisticRegression()
reg_log.fit(X_train, y_train)
y_pred = reg_log.predict(X_validation)


# Random forest decision tree
# reg_rf = RandomForestClassifier()
# reg_rf.fit(X_train, y_train)
# y_pred = reg_rf.predict(X_validation)


# Support vector machine
# reg_svc = SVC()
# reg_svc.fit(X_train, y_train)
# y_pred = reg_svc.predict(X_validation)


# Nearest neighbours
# reg_knn = KNeighborsClassifier()
# reg_knn.fit(X_train, y_train)
# y_pred = reg_knn.predict(X_validation)


print(metrics.classification_report(y_validation, y_pred))
