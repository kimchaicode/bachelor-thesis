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
ax = plt.gca()


X_original = np.array([row[:len(row) - 1] for row in data])
y_original = np.array([row[len(row) - 1] for row in data])


# sample_sizes = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
sample_size = 35000
# hidden_layer_size = Config.num_input_nodes * 2
# hidden_layers = (Config.num_input_nodes,Config.num_input_nodes,)
batch_sizes = [
    100,
    200, # default
    500,
    1000,
    5000
]

for (index, batch_size) in enumerate(batch_sizes):
    print(f"Training {batch_size}...")

    number_of_samples = sample_size * Config.max_agents
    X = X_original[:number_of_samples]
    y = y_original[:number_of_samples]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train = resample(X_train, y_train)

    model = train_model(X_train, y_train, batch_size=batch_size)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0.0))

    RocCurveDisplay.from_estimator(model, X_test, y_test, name=str(batch_size), ax=ax)

plt.show()
