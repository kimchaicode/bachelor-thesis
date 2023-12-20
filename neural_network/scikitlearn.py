# TODO After we have a working solution, we can compare all classifiers from
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import sys
sys.path.append("../config")

from config import Config

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from resampling import resample


# load data in a program from a text file
data = np.genfromtxt("../generator/results/agents.data", delimiter=',', dtype=int)

# TODO: Optimize splitting
X = np.array([row[:len(row) - 1] for row in data])
y = np.array([row[len(row) - 1] for row in data])



X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

X_train, y_train = resample(X_train, y_train)


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

print(classification_report(y_validation, y_pred, zero_division=0.0))

graph = RocCurveDisplay.from_estimator(model, X_validation, y_validation)
plt.show()
