class Config:
    # Allow for variable configuration of agents and test nodes with these parameters
    min_agents, max_agents = 10, 10
    min_test_nodes, max_test_nodes = 5, 5
    # Later: 1-5% infected
    min_infected, max_infected = 1, 1

    # we generate a row per agent, because we do binary classification on the infection status of each agent
    # in total there will be number_of_samples_per_agent * max_agents rows
    number_of_samples_per_agent = 50000

    num_input_nodes = max_agents * max_test_nodes + max_test_nodes + 1
    num_hidden_nodes = num_input_nodes

