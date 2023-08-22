# Question:
# Why is there no need to use nn.Softmax() in the output layer for a neural net when using nn.CrossEntropyLoss as a loss function
# Source: https://stackoverflow.com/questions/58122505/suppress-use-of-softmax-in-crossentropyloss-for-pytorch-neural-net
# Answer: nn.CrossEntryLoss is the combination of nn.LogSoftmax() and nn.NLLLoss()
# NLLLOSS: The negative log likelihood loss. It is useful to train a classification problem with C classes.
# BCELoss: Binary cross entry of predicted class.

# Hyperparemeters we can adjust
# 1. Learning rate
# 2. Number of epochs
# 3. Batch size
# 4. Dropout Rate
# 5. Pruning
# 6. Regularization Strength
#    L1, L2
# 7. Optimization Algorithm
#    SGD, Adam
# 8. Model architecture
#    Number of nodes in hidden layer

# 1. Learning rate
# too small => may result in a long training process that could get stuck
# too large => learning a sub-optimal set of weights too fast or unstable training process
# The learning rate may be the most important hyperparameter when configuring your neural network.

# 4. Dropout Rate
# prevent overfitting by randomly deactivating a fraction of neurons during each forward pass of the training phase

# 7. Optimization Algorithm
# See https://medium.com/geekculture/a-2021-guide-to-improving-cnns-optimizers-adam-vs-sgd-495848ac6008
# "Adam converges faster, SGD generalizes better than Adam and thus results in improved final performance"

# 8. Model architecture
# According to https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw/1097#1097
# > One hidden layer is sufficient for the large majority of problems.
# > The situations in which performance improves with a second (or third, etc.) hidden layer are very few.
# > the optimal size of the hidden layer is usually between the size of the input and size of the output layers
# > the number of neurons in that layer is the mean of the neurons in the input and output layers.

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryConfusionMatrix, BinaryF1Score

from data import agents

# From generator.py
max_agents = 10
max_test_nodes = 5

# input is of the following form: (test assignment graph, test result vector, agent identifier)
# size of test assignment graph = agents.max_agents * agents.max_test_nodes
# size of test result vector = agents.max_test_nodes
# size of agent identifier = 1
num_input_nodes = max_agents * max_test_nodes + max_test_nodes + 1
num_hidden_nodes = round(num_input_nodes / 2)

# 1. Set up the neural network with layers and activation function
class NeuralNetwork(nn.Sequential):
    def __init__(self):
        super(NeuralNetwork, self).__init__(
            nn.Linear(num_input_nodes, num_hidden_nodes),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(num_hidden_nodes, 1),
            nn.Sigmoid()
        )

# 2. Load the training data
# TODO: Upload the agents.data(textfile) from generator.py into main.py
training_inputs, validation_inputs = agents.get_datasets("./data/agents.data")

# We have 150 rows in training_inputs. With this batch_size we will go through all of the data in 2 epochs.
# Question: Is that okay or not?
# batch-size = 75 is okay
batch_size = 75
training_loader = DataLoader(dataset=training_inputs, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=validation_inputs, batch_size=len(validation_inputs), shuffle=True)


# 3. Instantiate the network, the loss function and the optimizer
# From Avik : hidden layer should be same size
net = NeuralNetwork()

# Why this specific loss function? => See above.
loss_function = nn.BCELoss()

learning_rate = 0.001
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


# 5. Train the neural network
num_epochs = 100

f1_score = BinaryF1Score()

train_loss = []
train_accuracy = []
train_f1_scores = []
validation_loss = []
validation_accuracy = []
validation_f1_scores = []


for epoch in range(num_epochs):

    for i, (X_batch, y_batch) in enumerate(training_loader):

        net.train()           # Put the network into training mode

        y_batch = y_batch.unsqueeze(1)

        y_pred = net(X_batch)  # Do the forward pass
        loss = loss_function(y_pred, y_batch) # Calculate the loss

        optimizer.zero_grad() # Clear off the gradients from any past operation
        loss.backward()       # Calculate the gradients with help of back propagation
        optimizer.step()      # Ask the optimizer to adjust the parameters based on the gradients

        print ('Epoch %d/%d, Iteration %d/%d, Loss: %.4f'
               %(epoch+1, num_epochs, i+1, len(training_inputs)//batch_size, loss.data.item()))

    net.eval()                 # Put the network into evaluation mode

    # Evaluate accuracy at end of each epoch
    # Record the training loss and accuracy
    accuracy = (y_pred.round() == y_batch).float().mean()
    train_accuracy.append(accuracy)
    train_loss.append(loss.data.item())
    train_f1_scores.append(f1_score(y_pred, y_batch))

    # How did we do on the validation set (the unseen set)
    # Record the correct predictions for validation data
    X_val, y_val = next(iter(validation_loader))
    y_val = y_val.unsqueeze(1)

    y_pred_val = net(X_val)
    loss = loss_function(y_pred_val, y_val)
    accuracy = (y_pred_val.round() == y_val).float().mean()

    validation_accuracy.append(accuracy)
    validation_loss.append(loss.data.item())
    validation_f1_scores.append(f1_score(y_pred_val, y_val))


# Create a confusion matrix after training on the validation set
confusion_matrix = BinaryConfusionMatrix()

predictions = []
targets = []

for inputs, labels in validation_loader:
    y_pred = net(inputs)

    predictions.extend(y_pred)
    targets.extend(labels.data)

confusion_matrix.update(torch.tensor(predictions), torch.tensor(targets))
confusion_matrix.plot()


fig = plt.figure(figsize=(12, 8))
plt.plot(train_accuracy, label='train accuracy')
plt.plot(validation_accuracy, label='validation accuracy')
plt.title("Train and Validation Accuracy")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(train_loss, label='train loss')
plt.plot(validation_loss, label='validation loss')
plt.title("Train and Validation Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.plot(train_f1_scores, label='train f1')
plt.plot(validation_f1_scores, label='validation f1')
plt.title("F1 Score")
plt.legend()
plt.show()
