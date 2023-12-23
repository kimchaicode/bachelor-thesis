from config import Config

from sklearn.neural_network import MLPClassifier

# Multi-layer perceptron
# input is of the following form: (test assignment graph, test result vector, agent identifier)
# size of test assignment graph = agents.max_agents * agents.max_test_nodes
# size of test result vector = agents.max_test_nodes
# size of agent identifier = 1
def train_model(X_train, y_train, hidden_layers=(Config.num_input_nodes,Config.num_input_nodes,), learning_rate=0.001):
    model = MLPClassifier(solver='adam', hidden_layer_sizes=hidden_layers, learning_rate_init=learning_rate, max_iter=500, early_stopping=True, verbose=True)
    model.fit(X_train, y_train)

    return model
