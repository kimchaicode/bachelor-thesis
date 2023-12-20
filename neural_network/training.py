from config import Config

from sklearn.neural_network import MLPClassifier

def train_model(X_train, y_train):
    # Multi-layer perceptron
    # input is of the following form: (test assignment graph, test result vector, agent identifier)
    # size of test assignment graph = agents.max_agents * agents.max_test_nodes
    # size of test result vector = agents.max_test_nodes
    # size of agent identifier = 1
    num_input_nodes = Config.max_agents * Config.max_test_nodes + Config.max_test_nodes + 1
    num_hidden_nodes = round(num_input_nodes / 2)

    model = MLPClassifier(solver='adam', hidden_layer_sizes=(num_hidden_nodes,), early_stopping=True, verbose=True)
    model.fit(X_train, y_train)

    return model
