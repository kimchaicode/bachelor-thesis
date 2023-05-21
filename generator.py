import random
import numpy as np

number_of_agents = 7
number_of_test_nodes = 5
matrix_size = number_of_agents + number_of_test_nodes

graph_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int8)

for x in range(0, number_of_agents):
    for y in range(number_of_agents, number_of_agents + number_of_test_nodes):
        rand = random.randint(0,1)
        graph_matrix[x][y] = rand
        graph_matrix[y][x] = rand

np.savetxt('./results/1', graph_matrix, fmt='%i')

print(graph_matrix)

