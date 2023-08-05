import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

class MultiClassDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        class_index = len(row) - 1
        return (
            # "Agent features" = test assignment graph + test result vector as one array
            row[0:class_index].astype(np.float32),
            # Class number
            # In our case we transformed the agent infection vector to a class as binary,
            #   e.g if agents 1 and 3 are infected the class number is 5 (101 as binary).
            row[class_index].astype(int)
        )

    def __len__(self):
        return self.data.shape[0]


def get_datasets(file_path, train_ratio=0.80):

    data = np.genfromtxt(file_path, delimiter=',', dtype=int)

    # TODO Split into training data and test data via `train_ratio`
    # train_df = data.sample(frac=train_ratio, random_state=3)
    # test_df = data.loc[~data.index.isin(train_df.index), :]

    # return IrisDataset(train_df), IrisDataset(test_df)

    return MultiClassDataset(data)

