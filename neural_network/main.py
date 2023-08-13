# Question:
# Why is there no need to use nn.Softmax() in the output layer for a neural net when using nn.CrossEntropyLoss as a loss function
# Source: https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net
# Answer: nn.CrossEntryLoss is the combination of nn.LogSoftmax() and nn.NLLLoss()
# NLLLOSS: The negative log likelihood loss. It is useful to train a classification problem with C classes.
# BCELoss: Binary cross entry of predicted class.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score

from data import agents


# From generator.py
max_agents = 10
max_test_nodes = 5

# input is of the following form: (test assignment graph, test result vector, agent identifier)
# size of test assignment graph = agents.max_agents * agents.max_test_nodes
# size of test result vector = agents.max_test_nodes
# size of agent identifier = 1
num_input_nodes = max_agents * max_test_nodes + max_test_nodes + 1

# 1. Set up the neural network with layers and activation function
class NeuralNetwork(nn.Sequential):
    def __init__(self):
        super(NeuralNetwork, self).__init__(
            nn.Linear(num_input_nodes, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

# 2. Load the training data
# TODO: Upload the agents.data(textfile) from generator.py into main.py
training_inputs = agents.get_datasets("./data/agents.data")

# We have 150 rows in training_inputs. With this batch_size we will go through all of the data in 2 epochs.
# Question: Is that okay or not?
# batch-size = 75 is okay
batch_size = 75
training_loader = DataLoader(dataset=training_inputs, batch_size=batch_size, shuffle=True)

# 3. Instantiate the network, the loss function and the optimizer
# From Avik : hidden layer should be same size
net = NeuralNetwork()

# Why this specific loss function? => See above.
loss_function = nn.BCELoss()

# TODO Why this specific learning rate, optimizer?
learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# 5. Train the neural network, 500 epochs
num_epochs = 500

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []

f1_scores = []
f1_score = BinaryF1Score()

for epoch in range(num_epochs):

    train_correct = 0
    train_total = 0

    for i, (X_batch, y_batch) in enumerate(training_loader):

        # net.train()           # Put the network into training mode
        y_batch = y_batch.unsqueeze(1)

        y_pred = net(X_batch)  # Do the forward pass
        loss = loss_function(y_pred, y_batch) # Calculate the loss

        optimizer.zero_grad() # Clear off the gradients from any past operation
        loss.backward()       # Calculate the gradients with help of back propagation
        optimizer.step()      # Ask the optimizer to adjust the parameters based on the gradients

        print ('Epoch %d/%d, Iteration %d/%d, Loss: %.4f'
               %(epoch+1, num_epochs, i+1, len(training_inputs)//batch_size, loss.data.item()))

    # net.eval()                 # Put the network into evaluation mode

    # Evaluate accuracy at end of each epoch
    # Record the training loss and accuracy
    train_loss.append(loss.data.item())
    accuracy = (y_pred.round() == y_batch).float().mean()
    train_accuracy.append(accuracy)

    f1_scores.append(f1_score(y_pred, y_batch))

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
plt.plot(train_accuracy, label='train accuracy')
# plt.plot(test_accuracy, label='test accuracy')
plt.title("Train and Test Accuracy")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='train loss')
# plt.plot(test_loss, label='test loss')
plt.title("Train and Test Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(f1_scores, label='f1 score')
# plt.plot(test_accuracy, label='test accuracy')
plt.title("F1 Score")
plt.legend()
plt.show()
