# TODO After we have a working solution, we can compare all classifiers from
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
import sys
sys.path.append("../config")

from config import Config

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split

from resampling import resample
from training import train_model


# load data in a program from a text file
data = np.genfromtxt("../generator/results/agents.data", delimiter=',', dtype=int)

# TODO: Optimize splitting
X = np.array([row[:len(row) - 1] for row in data])
y = np.array([row[len(row) - 1] for row in data])


X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)
X_train, y_train = resample(X_train, y_train)

model = train_model(X_train, y_train)

# y_pred = model.predict(X_validation)
# print(classification_report(y_validation, y_pred, zero_division=0.0))

graph = RocCurveDisplay.from_estimator(model, X_validation, y_validation)
plt.show()
