import random
import numpy as np

number_of_agents = 7
number_of_test_nodes = 5
matrix_size = number_of_agents + number_of_test_nodes

graph_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int8)

# Optimzation: Do not store whole matrix, we're only interested in the relation of agents to test nodes (top right part of matrix)
for x in range(0, number_of_agents):
    for y in range(number_of_agents, number_of_agents + number_of_test_nodes):
        rand = random.randint(0,1)
        graph_matrix[x][y] = rand
        graph_matrix[y][x] = rand

# Optimization: Store as binary for smaller size
np.savetxt('./results/1', graph_matrix, fmt='%i')

print(graph_matrix)

