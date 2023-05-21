import numpy as np

number_of_agents = 7
number_of_test_nodes = 5

graph_matrix = np.random.randint(2, size=(number_of_agents, number_of_test_nodes))

# Optimization: Store as binary for smaller size
np.savetxt('./results/1', graph_matrix, fmt='%i')

print(graph_matrix)

