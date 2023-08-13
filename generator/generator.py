# Write the data in the following way into the same file linearly, i.e. 1 row per data point
# graph_matrix, test_result_vector, infection_vector as class number (binary)
# Example: 0,1,1,1,0,1, 1,2,               1
#         |graph_matrix|test_result_vector|agent identifier

# TODO: Questions
# 1. How many agents do we want? like 1000, increase agents number, try 100 at first
# 2. How many test nodes do we want?
# 3. Is it okay to have more test nodes than agents?
# 4. Do we need any more extra rules for the test assignment graph?
# 5. How many samples do we need? too low, more than 500.

import random

import numpy as np

# Allow for variable configuration of agents and test nodes with these parameters
min_agents, max_agents = 10, 10
min_test_nodes, max_test_nodes = 5, 5
# Later: 1-5% infected
min_infected, max_infected = 1, 1

# we generate a row per agent, because we do binary classification on the infection status of each agent
# in total there will be number_of_samples_per_agent * max_agents rows
number_of_samples_per_agent = 150

with open("./results/agents.data", "w") as file:
    for i in range(number_of_samples_per_agent):
        # Random configuration for number of agents
        number_of_agents = random.randint(min_agents, max_agents)
        # Random configuration for test nodes, but do not allow more test nodes than agents
        number_of_test_nodes = random.randint(min_test_nodes, min([max_test_nodes, number_of_agents]))

        # Because we have variable number of agents and test nodes, we need to pad the difference of
        # the current configuration to the maximum configuration with zeros
        graph_matrix = np.random.randint(2, size=(number_of_test_nodes, number_of_agents))
        graph_matrix = np.pad(graph_matrix, [(0, max_test_nodes - number_of_test_nodes), (0, max_agents - number_of_agents)], mode='constant', constant_values=0)

        number_of_infected = random.randint(min_infected, max_infected)
        # Create an array with a specific number of 1s (infected), and rest 0s
        # See https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
        infection_vector = np.array([1] * number_of_infected + [0] * (number_of_agents - number_of_infected)).reshape(number_of_agents, 1)
        np.random.shuffle(infection_vector)

        # Pad the infection_vector, so that we can calculate the dot product with the padded graph_matrix
        infection_vector = np.pad(infection_vector, [(0, max_agents - number_of_agents), (0, 0)], mode='constant', constant_values=0)
        test_result_vector = np.dot(graph_matrix, infection_vector)

        # For each agent, write one sample that has the agent as the identifier, and the infection status as the output class
        for agent_identifier in range(number_of_agents):
            infection_status = infection_vector[agent_identifier]

            # Output each sample in one row, comma-separated
            np.savetxt(file, graph_matrix, fmt='%i', delimiter=",", newline=",")
            np.savetxt(file, test_result_vector, fmt='%i', delimiter=",", newline=",")
            np.savetxt(file, [agent_identifier], fmt='%i', delimiter=",", newline=",")
            np.savetxt(file, infection_status, fmt='%i')

