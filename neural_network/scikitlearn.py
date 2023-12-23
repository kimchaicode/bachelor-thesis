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
hidden_layers = (Config.num_input_nodes,Config.num_input_nodes,)
learning_rates = [
    0.0005,
    0.00075,
    0.001, # default
    0.00125,
    0.0015
]

for (index, learning_rate) in enumerate(learning_rates):
    print(f"Training {learning_rate}...")

    number_of_samples = sample_size * Config.max_agents
    X = X_original[:number_of_samples]
    y = y_original[:number_of_samples]

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)
    X_train, y_train = resample(X_train, y_train)

    model = train_model(X_train, y_train, learning_rate=learning_rate)

    y_pred = model.predict(X_validation)
    print(classification_report(y_validation, y_pred, zero_division=0.0))

    RocCurveDisplay.from_estimator(model, X_validation, y_validation, name=str(learning_rate), ax=ax)

plt.show()
