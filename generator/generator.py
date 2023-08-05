# Write the data in the following way into the same file linearly, i.e. 1 row per data point
# graph_matrix, test_result_vector, infection_vector as class number (binary)
# Example: 0,1,1,1,0,1, 1,2,               5
#         |graph_matrix|test_result_vector|class_number (infection vector as binary number)|

import numpy as np

number_of_agents = 3
number_of_test_nodes = 2

number_of_data_sets = 150

with open("./results/agents.data", "w") as file:
    for i in range(number_of_data_sets):
        graph_matrix = np.random.randint(2, size=(number_of_test_nodes, number_of_agents))
        infection_vector = np.random.randint(2, size=(number_of_agents, 1))
        infection_vector_as_string = "".join(map(str, infection_vector.flatten()))
        infection_vector_as_binary_number = int(infection_vector_as_string, 2)
        test_result_vector = np.dot(graph_matrix, infection_vector)

        np.savetxt(file, graph_matrix, fmt='%i', delimiter=",", newline=",")
        np.savetxt(file, test_result_vector, fmt='%i', delimiter=",", newline=",")
        np.savetxt(file, [infection_vector_as_binary_number], fmt='%i')

