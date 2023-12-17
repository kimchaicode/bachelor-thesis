# TODO After we have a working solution, we can compare all classifiers from
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import sys
sys.path.append("../config")

from config import Config

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# load data in a program from a text file
data = np.genfromtxt("../generator/results/agents.data", delimiter=',', dtype=int)

# TODO: Optimize splitting
X = np.array([row[:len(row) - 1] for row in data])
y = np.array([row[len(row) - 1] for row in data])

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

# Apply over- and undersampling
print("Destribution before...")
print(Counter(y_train))

# - [ ] Check default `sampling_strategy` parameters
oversample = SMOTE(sampling_strategy=0.5)
X_train, y_train = oversample.fit_resample(X_train, y_train)

print("Destribution after oversampling...")
print(Counter(y_train))

undersample = RandomUnderSampler()
X_train, y_train = undersample.fit_resample(X_train, y_train)

# - [ ] Print class distributions/ Counter(y) to check if resampling worked
print("Destribution after undersampling...")
print(Counter(y_train))


# Try different classifiers...
# ============================

# Logistic regression
# reg_log = LogisticRegression()
# reg_log.fit(X_train, y_train)
# y_pred = reg_log.predict(X_validation)


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


# Decision tree
# reg_dt = DecisionTreeClassifier()
# reg_dt.fit(X_train, y_train)
# y_pred = reg_dt.predict(X_validation)


# Multi-layer perceptron
# input is of the following form: (test assignment graph, test result vector, agent identifier)
# size of test assignment graph = agents.max_agents * agents.max_test_nodes
# size of test result vector = agents.max_test_nodes
# size of agent identifier = 1
num_input_nodes = Config.max_agents * Config.max_test_nodes + Config.max_test_nodes + 1
num_hidden_nodes = round(num_input_nodes / 2)

model = MLPClassifier(solver='adam', hidden_layer_sizes=(num_hidden_nodes,), early_stopping=True, verbose=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)

print(metrics.classification_report(y_validation, y_pred, zero_division=0.0))


# Try different values for k_neighbors and print the ROC area under curve metric
# k_values = [1, 2, 3, 4, 5, 6, 7]
# for k in k_values:
#     steps = [('over', SMOTE(sampling_strategy=0.15, k_neighbors=k)), ('under', RandomUnderSampler(sampling_strategy=0.5)), ('model', DecisionTreeClassifier())]
#     pipeline = Pipeline(steps=steps)

#     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#     scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
#     score = np.mean(scores)
#     print('> k=%d, Mean ROC AUC: %.3f' % (k, score))
