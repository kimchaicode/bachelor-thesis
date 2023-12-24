from config import Config

from sklearn.neural_network import MLPClassifier

# Multi-layer perceptron
# input is of the following form: (test assignment graph, test result vector, agent identifier)
# size of test assignment graph = agents.max_agents * agents.max_test_nodes
# size of test result vector = agents.max_test_nodes
# size of agent identifier = 1
def train_model(X_train, y_train):
    model = MLPClassifier(solver='adam', hidden_layer_sizes=(Config.num_input_nodes,Config.num_input_nodes,), max_iter=500, early_stopping=True, verbose=True)
    model.fit(X_train, y_train)

    return model
