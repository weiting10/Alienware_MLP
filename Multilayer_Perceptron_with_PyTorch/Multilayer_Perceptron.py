import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

ROOT = '.data'

train_data = datasets.MNIST(root=ROOT, train=True, download=True)

# normalize data: make data to have mean of zero and a standard deviation of one.
mean = train_data.data.float().mean() / 255
std = train_data.data.float().std() / 255

print(f'Calculated mean: {mean}')
print(f'Calculated std: {std}')

# Data augmentation: manipulating the available training data in a way that artifically creates more training
# examples (randomely rotating, adding padding around the image) augmented data will be transformed to a tensor and
# normalize

train_transforms = transforms.Compose([
    transforms.RandomRotation(5, fill=(0,)),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[mean], std=[std])
])

# Load the train and test data with the relevant defined transforms
train_data = datasets.MNIST(root=ROOT,
                            train=True,
                            download=True,
                            transform=train_transforms)

test_data = datasets.MNIST(root=ROOT,
                           train=False,
                           download=True,
                           transform=test_transforms)
# Length of datasets
print(f'Number of training examples: {len(train_data)}')
print(f'Number of testing examples: {len(test_data)}')


# Sample image visualization
def plot_images(N_IMAGES, input_data):  # input_data = train_data or test_data or va
    images = [image for image, label in [input_data[i] for i in range(N_IMAGES)]]
    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure()
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(images[i].view(28, 28).cpu().numpy(), cmap='bone')
        ax.axis('off')


# Creating a validation data for a proxy test to check how model performs
VALID_RATIO = 0.9

num_train_examples = int(len(train_data) * VALID_RATIO)
num_valid_examples = len(train_data) - num_train_examples

# Take a random 10% of the training set to use as a validation set
train_data, valid_data = data.random_split(train_data,
                                           [num_train_examples, num_valid_examples])

# Check number of examples for each portions
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# Replace the validation set's transform by overwriting it with previously built test transforms
valid_data = copy.deepcopy(valid_data)  # To prevent changing of default transforms of other training data
valid_data.dataset.transform = test_transforms

# Data loader to for each set (training/valid/test) with batches to feed iteratively to the model
# only train data is shuffled
BATCH_SIZE = 64

train_iterator = data.DataLoader(train_data,
                                 shuffle=True,
                                 batch_size=BATCH_SIZE)

valid_iterator = data.DataLoader(valid_data,
                                 batch_size=BATCH_SIZE)

test_iterator = data.DataLoader(test_data,
                                batch_size=BATCH_SIZE)


class Multilayer_Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forwad_propagation(self, x):  # x:  input tensor to the network

        # x = [batch_size, height, width]
        batch_size = x.shape[0]
        # transform to
        x = x.view(batch_size, -1)  # reshape the tensor to x = [batch size, height * width] -
        # 1 is when not sure about number of rows

        # First neural layer
        nn_layer_1 = self.input_fc(x)
        nn_layer_act_func_1 = F.relu(nn_layer_1)

        # Second or Hidden neural layer
        nn_layer_2 = self.hidden_fc(nn_layer_act_func_1)
        nn_layer_act_func_2 = F.relu(nn_layer_2)

        # Output layer or prediction layer
        # y_predict_layer = [batch_size_output_dim]
        y_predict_layer = self.output_fc(nn_layer_act_func_2)

        return y_predict_layer, nn_layer_act_func_2


# Define Multilayer perceptron model by creating an instance of it and setting the correct input and output dimensions.

INPUT_DIM = 28 * 28
OUTPUT_DIM = 10

model = Multilayer_Perceptron(INPUT_DIM, OUTPUT_DIM)


# Calculate the number of trainable parameters (weights and biases)
def count_parameters(mlp_model):
    return sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)


# Calculate and print number of trainable parameters
print(f'The model has {count_parameters(model):,} trainable parameters')

# Optimizer
optimizer = optim.Adam(model.parameters())
# Cost function
criterion = nn.CrossEntropyLoss()

# define device to put model and data, by defult it is GPU or else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Put the model and cost function in the defined device
model = model.to(device)
criterion = criterion.to(device)


# Calculate the accuracy of the model
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


'''
Define the training loop function for the model
The training loop will perform following tasks:
    * put the model into train mode
    * iterate over the data loader, returning batches of (image, label)
    * place the batch on to GPU, if not available on CPU
    * clear the gradients calculated from the last batch
    * pass a batch of images, x, through to model to get predictions, y_pred
    * calculate the loss between the predictions and the actual labels
    * calculate the accuracy between our predictions and the actual labels
    * calculate the gradients of each parameter
    * update the parameters by taking an optimizer step
    * update metrics
'''


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred, _ = model.forwad_propagation(x)

        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


'''
The evaluation loop is similar to the training loop bu serves different purposes performing following:
    * put to model into evaluation mode with model.eval()
    * wrap the iterations inside a with torch.no_grad() to make sure gradients are not calculated in evaluation step
    * do not calculate gradients as we are not updating parameters
    * do not take an optimizer step as we are not calculating gradients
'''


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred, A = model.forwad_propagation(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# Define a function to tell how long an epoch took
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Model training
best_valid_loss = float('inf')

EPOCHS = 10

for epoch in range(EPOCHS):

    start_time = time.monotonic()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion, device)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)

    # Make sure always saves the set of parameters that has the best validation loss (validation accuracy)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')

    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Valid Loss: {valid_loss:.3f} |  Valid Acc: {valid_acc * 100:.2f}%')

# Test the model
# Afterwards, load the parameters of the model that achieved the best validation loss
# Then use this to evaluate our model on the test set.

model.load_state_dict(torch.load('tut1-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


# Examining the model with simple exploratory
# This function will return input image and model prediction output with ground truth
def get_predictions(model, iterator, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred, _ = model.forwad_propagation(x)

            y_prob = F.softmax(y_pred, dim=-1)
            top_pred = y_prob.argmax(1, keepdim=True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs


# Getting the predictions
images, labels, probs = get_predictions(model, test_iterator, device)
# It can get these predictions or prediction labels  and, by taking the index of the highest predicted probability.
pred_labels = torch.argmax(probs, 1)


# Develop confusion matrix from the actual labels and the predicted labels
def plot_confusion_matrix(labels, pred_labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(10))
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.savefig('confusion_matrix.png')
    plt.show()


plot_confusion_matrix(labels, pred_labels)

# Check whether predicted labels and actual labesl matches or no
corrects = torch.eq(labels, pred_labels)
# FInd out incorrectly classified examples into an array
incorrect_examples = []

for image, label, prob, correct in zip(images, labels, probs, corrects):
    if not correct:
        incorrect_examples.append((image, label, prob))

incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)


# Plot the incorrectly predicted images along with how confident they were on the actual label
# Then see how confident they were at the incorrect label.

def plot_most_incorrect(incorrect, n_images):
    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        image, true_label, probs = incorrect[i]
        true_prob = probs[true_label]
        incorrect_prob, incorrect_label = torch.max(probs, dim=0)
        ax.imshow(image.view(28, 28).cpu().numpy(), cmap='bone')
        ax.set_title(f'true label: {true_label} ({true_prob:.3f})\n' \
                     f'pred label: {incorrect_label} ({incorrect_prob:.3f})')
        ax.axis('off')
    fig.subplots_adjust(hspace=0.5)
    plt.savefig('most_incorrect.png')
    plt.show()


# Try with 25 incorrectly classified image compared with ground truth and see how confident are the predections
N_IMAGES = 25
plot_most_incorrect(incorrect_examples, N_IMAGES)


# For more understanding it can get the output and intermediate representations from the model and try to visualize them
def get_representations(model, iterator, device):
    model.eval()

    outputs = []
    intermediates = []
    labels = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)

            y_pred, h = model.forwad_propagation(x)

            outputs.append(y_pred.cpu())
            intermediates.append(h.cpu())
            labels.append(y)

    outputs = torch.cat(outputs, dim=0)
    intermediates = torch.cat(intermediates, dim=0)
    labels = torch.cat(labels, dim=0)

    return outputs, intermediates, labels


# Get representations
outputs, intermediates, labels = get_representations(model, train_iterator, device)


# Since data to be visualized are in high dimensional space (10 and 100)
# Therefore us principal component analysis (PCA)  to bring to 2-dimensional data
# Calculate PCS first
def get_pca(data, n_components=2):
    pca = decomposition.PCA()
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    return pca_data


# Plot representations after dimensionality reduction
def plot_representations(data, labels, type, n_images=None):
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10')
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles=handles, labels=labels)
    if (type == 'output'):
        plt.savefig('output_representations.png')
        plt.show()
    elif (type == 'intermediates'):
        plt.savefig('intermediates_representations.png')
        plt.show()


# Output representations from the ten dimensional output layer, reduced down to two dimensions (10-dim)
output_pca_data = get_pca(outputs)
plot_representations(output_pca_data, labels, 'output')

# Plot outputs of the second hidden layer, reduced down to two dimensions (100-dim)
intermediate_pca_data = get_pca(intermediates)
plot_representations(intermediate_pca_data, labels, 'intermediates')


# Finally, it can plot the weights in the first layer of the model
# Since neural networks follows hierarchical learning it is meaningful to visualize weights of 1st layer
def plot_weights(weights, n_weights):
    rows = int(np.sqrt(n_weights))
    cols = int(np.sqrt(n_weights))

    fig = plt.figure(figsize=(20, 10))
    for i in range(rows * cols):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(weights[i].view(28, 28).cpu().numpy(), cmap='bone')
        ax.axis('off')

    plt.savefig('weights_visualization.png')
    plt.show()


# Plotting 25 weights
N_WEIGHTS = 25
weights = model.input_fc.weight.data
plot_weights(weights, N_WEIGHTS)
