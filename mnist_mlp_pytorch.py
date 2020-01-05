#The process will be broken down into the following steps:
"""
    1.Load and visualize the data
    2.Define a neural network
    3.Train the model
    4.Evaluate the performance of our trained model on a test dataset!
"""

# import libraries
import torch
import numpy as np
from torch import optim

"""
Downloading may take a few moments, and you should see your progress as the data is loading.
You may also choose to change the batch_size if you want to load more data at a time.
"""

from torchvision import datasets
import torchvision.transforms as transforms

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers)

"""
Visualize a Batch of Training Data

The first step in a classification task is to take a look at the data, make sure it is loaded in correctly,
then make any initial observations about patterns in that data.

"""
import matplotlib.pyplot as plt
%matplotlib inline
    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))
  
"""
View an Image in More Detail
"""
img = np.squeeze(images[1])

fig = plt.figure(figsize = (12,12)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
width, height = img.shape
thresh = img.max()/2.5
for x in range(width):
    for y in range(height):
        val = round(img[x][y],2) if img[x][y] !=0 else 0
        ax.annotate(str(val), xy=(y,x),
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
"""
Define the Network Architecture

The architecture will be responsible for seeing as input a 784-dim Tensor of pixel values for each image,
and producing a Tensor of length 10 (our number of classes) that indicates the class scores for an input image.
This particular example uses two hidden layers and dropout to avoid overfitting.
"""
    
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear layer (784 -> 1 hidden node)
        h1,h2 = 512,512
        self.fc1 = nn.Linear(28 * 28, h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2,10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)


"""
Specify Loss Function and Optimizer

we use cross-entropy loss for classification. 
you can see that PyTorch's cross entropy function applies a softmax funtion 
to the output layer and then calculates the log loss.


"""
# specify loss function
criterion = nn.CrossEntropyLoss()

# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)


