import math

import numpy as np

from config.config import Config
from torch.utils.data.dataset import Dataset

number_of_samples = Config.number_of_samples_per_agent * Config.max_agents

class AgentDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        output_index = len(row) - 1
        return (
            # "Agent features" = test assignment graph + test result vector as one array
            row[0:output_index].astype(np.float32),
            # Class number = infection status
            row[output_index].astype(np.float32)
        )

    def __len__(self):
        return self.data.shape[0]


def get_datasets(file_path, train_ratio=0.80):

    data = np.genfromtxt(file_path, delimiter=',', dtype=int)
    np.random.shuffle(data)

    split_index = math.floor(number_of_samples * train_ratio)
    training_data, validation_data = data[:split_index,:], data[split_index:,:]

    return AgentDataset(training_data), AgentDataset(validation_data)

