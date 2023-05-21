import numpy as np

number_of_agents = 7
number_of_test_nodes = 5

graph_matrix = np.random.randint(2, size=(number_of_test_nodes, number_of_agents))

# Optimization: Store in one file
# Optimization: Store as binary for smaller size
np.savetxt('./results/1', graph_matrix, fmt='%i')
print(graph_matrix)

infection_vector = np.random.randint(2, size=(number_of_agents, 1))
np.savetxt('./results/1.infection', infection_vector, fmt='%i')
print(infection_vector)

test_result_vector = np.dot(graph_matrix, infection_vector)
np.savetxt('./results/1.test_result', test_result_vector, fmt='%i')
print(test_result_vector)
