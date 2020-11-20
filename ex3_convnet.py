import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# --------------------------------
# Hyper-parameters
# --------------------------------
input_size = 3
num_classes = 10
hidden_size = [128, 512, 512, 512, 512]
num_epochs = 20
batch_size = 200
learning_rate = 2e-3
learning_rate_decay = 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
# norm_layer = None
norm_layer = 'BN'
print(hidden_size)

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------
#################################################################################
# TODO: Q3.a Choose the right data augmentation transforms with the right       #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
norm_transform = transforms.Compose(data_aug_transforms + [transforms.ToTensor(),
                                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                           ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                             train=True,
                                             transform=norm_transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                            train=False,
                                            transform=test_transform
                                            )

# -------------------------------------------------
# Prepare the training and validation splits
# -------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)

# -------------------------------------------------
# Data loader
# -------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# -------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
# -------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # composes the convolution blocks
        for i_block, out_channels in enumerate(hidden_layers):
            in_channels = input_size if i_block == 0 else hidden_size[i_block - 1]
            conv_layers = [
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=(3, 3), stride=1, padding=1),
                nn.MaxPool2d(2),
                nn.ReLU()
            ]
            # eventually adds a normalization layer into the convolution block
            if norm_layer:
                conv_layers.insert(1, nn.BatchNorm2d(out_channels))
            conv_block = nn.Sequential(*conv_layers)
            layers += [conv_block]

        # adds the classification block
        classification_layers = [
            nn.Flatten(),
            nn.Linear(hidden_size[-1], num_classes)
        ]
        classification_block = nn.Sequential(*classification_layers)
        layers += [classification_block]

        # packs all the convolutional blocks and the classification block into a single sequential container
        self.layers = nn.Sequential(*layers)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # passes the input set to the sequence of convolutional blocks and then to the classification block
        out = self.layers(x)

        # adds the softmax only if we don't have to calculate the loss
        if not self.training:
            out = F.softmax(out, dim=-1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out


# -------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
# -------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # retrieves the number of parameters in the model
    model_sz = sum(parameter.numel() for parameter in model.parameters())
    print(f"Number of parameters in the model: {model_sz}")

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz


# -------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
# -------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image of stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # retrieves the weights (named also filters or kernels) of the very first convolutional layer
    # and normalizes all the values to be in [0, 1]
    filters = model.layers[0][0].weight.cpu().detach().numpy()
    filters = np.interp(filters, (filters.min(), filters.max()), (0, 1))

    # dynamically create a square plot
    filters_no = filters.shape[0]
    rows_no, columns_no = int(np.ceil(np.sqrt(filters_no))), int(np.ceil(np.sqrt(filters_no)))
    fig, axs = plt.subplots(rows_no, columns_no)

    # fills each subplot with a filter
    filter_i = 0
    for r in range(rows_no):
        for c in range(columns_no):
            ax = axs[r, c]
            # hides the labels
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # as the number of subplots can exceed the number of effective filters,
            # we have to check if we've not in a "blank" subplot
            if filter_i < filters_no:
                ax.imshow(filters[filter_i])
            filter_i += 1
    plt.show()
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


# ======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
# ======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
# --------------------------------------------------------------------------------------
model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

model.apply(weights_init)
# Print the model
print(model)
# Print model size
# ======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
# ======================================================================================
PrintModelSize(model)
# ======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
# ======================================================================================
VisualizeFilter(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

# Train the model
lr = learning_rate
total_step = len(train_loader)
loss_train = []
loss_val = []
best_accuracy = None
accuracy_val = []
best_model = type(model)(input_size, hidden_size, num_classes, norm_layer=norm_layer)  # get a new instance
# best_model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer)
for epoch in range(num_epochs):

    model.train()

    loss_iter = 0
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_iter += loss.item()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    loss_train.append(loss_iter / (len(train_loader) * batch_size))

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_iter = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss_iter += loss.item()

        loss_val.append(loss_iter / (len(val_loader) * batch_size))

        accuracy = 100 * correct / total
        accuracy_val.append(accuracy)
        print('Validation accuracy is: {} %'.format(accuracy))
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # the model with the best validation accuracy so-far (use best_model).          #
        #################################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # checks if we have found a better model
        if epoch == 0 or accuracy > np.max(accuracy_val):
            best_model = model
            # saves the model checkpoint
            torch.save(best_model.state_dict(), 'model.ckpt')
            if epoch > 0:
                print(
                    f"\tfound a new better model, with a validation accuracy of {accuracy}% "
                    f"({accuracy - np.max(accuracy_val)}% better)")

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()

plt.figure(2)
plt.plot(loss_train, 'r', label='Train loss')
plt.plot(loss_val, 'g', label='Val loss')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(accuracy_val, 'r', label='Val accuracy')
plt.legend()
plt.show()

#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# loads the weights from the file
best_model = torch.load('model.ckpt')

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Compute accuracy on the test set
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
