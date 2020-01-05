# MNIST-PyTorch
Classification of MNIST dataset using PyTorch
  Multi-Layer Perceptron, MNIST
  
  steps 1: Load and visualize the data
  
        2: Define a neural network
        
        3: Train the model
        
        4: Evaluate the performance of our model on test dataset!

Step 1: Loading and visualizing Data

    In this step we will use torchvision.dataset for downloading MNIST data set from torchvision and 
    then use matplotlib to visualize them.
  

step 2: Define a neural network
  
    The architecture will be responsible for seeing as input a 784-dim Tensor of pixel values for each image, and producing a     Tensor of length 10 (our number of classes) that indicates the class scores for an input image. This particular example uses two hidden layers and dropout to avoid overfitting.

setp 3: Train the Network

The steps for training/learning from a batch of data are described in the comments below:

    1.Clear the gradients of all optimized variables
    2.Forward pass: compute predicted outputs by passing inputs to the model
    3.Calculate the loss
    4.Backward pass: compute gradient of the loss with respect to model parameters
    5.Perform a single optimization step (parameter update)
    6.Update average training loss

step 4: Test the Trained Network

    Finally, we test our best model on previously unseen test data and evaluate it's performance. Testing on unseen data is a good way to check that our model generalizes well. It may also be useful to be granular in this analysis and take a look at how this model performs on each class as well as looking at its overall loss and accuracy.


    model.eval() will set all the layers in your model to evaluation mode. This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation!
