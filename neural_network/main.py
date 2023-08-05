# Question:
# Why is there no need to use nn.Softmax() in the output layer for a neural net when using nn.CrossEntropyLoss as a loss function
# Source: https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net
# Answer: nn.CrossEntryLoss is the combination of nn.LogSoftmax() and nn.NLLLoss()


# TODO Saturday
# 2. Generate more data, e.g. 150 samples
#    1. Update generator to generate simpler rows

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from data import agents

# 1. Set up the neural network with layers and activation function
class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


# 2. Load the training data
training_inputs = agents.get_datasets("./data/agents.data")

batch_size = 1
training_loader = DataLoader(dataset=training_inputs, batch_size=batch_size, shuffle=True)

# 3. Instantiate the network, the loss function and the optimizer
net = NeuralNetwork(8, 100, 50, 8)

# Why this specific loss function? => See above.
loss_function = nn.CrossEntropyLoss()

# TODO Why this specific learning rate, optimizer, (nesterov) momentum, and dampening?
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, nesterov=True, momentum=0.9, dampening=0)


# 5. Train the neural network, 500 epochs
num_epochs = 500

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

for epoch in range(num_epochs):

    train_correct = 0
    train_total = 0

    for i, (items, classes) in enumerate(training_loader):

        # Convert torch tensor to Variable
        items = Variable(items)
        classes = Variable(classes)

        # net.train()           # Put the network into training mode

        outputs = net(items)  # Do the forward pass
        loss = loss_function(outputs, classes) # Calculate the loss

        optimizer.zero_grad() # Clear off the gradients from any past operation
        loss.backward()       # Calculate the gradients with help of back propagation
        optimizer.step()      # Ask the optimizer to adjust the parameters based on the gradients

        # Record the correct predictions for training data
        train_total += classes.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == classes.data).sum()

        print ('Epoch %d/%d, Iteration %d/%d, Loss: %.4f'
               %(epoch+1, num_epochs, i+1, len(training_inputs)//batch_size, loss.data.item()))

    # net.eval()                 # Put the network into evaluation mode

    # Book keeping
    # Record the loss
    train_loss.append(loss.data.item())
    train_accuracy.append((100 * train_correct / train_total))

    # How did we do on the test set (the unseen set)
    # Record the correct predictions for test data
    # test_items = torch.FloatTensor(test_ds.data.values[:, 0:4])
    # test_classes = torch.LongTensor(test_ds.data.values[:, 4])

    # outputs = net(Variable(test_items))
    # loss = criterion(outputs, Variable(test_classes))
    # test_loss.append(loss.data.item())
    # _, predicted = torch.max(outputs.data, 1)
    # total = test_classes.size(0)
    # correct = (predicted == test_classes).sum()
    # test_accuracy.append((100 * correct / total))


fig = plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='train loss')
# plt.plot(test_loss, label='test loss')
plt.title("Train and Test Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(train_accuracy, label='train accuracy')
# plt.plot(test_accuracy, label='test accuracy')
plt.title("Train and Test Accuracy")
plt.legend()
plt.show()
