'''
1. We import the stuff we need: PyTorch to build the model, torchvision to handle our images.

2. We download our handwritten digits (0 to 9) from the MNIST dataset. We use DataLoader to easily handle this data.

3. We build our model ("brain") using PyTorch's Sequential function. It's three layers of neurons and we use ReLU to make them work well together.

4. We set up our problem-solving tools: CrossEntropyLoss to tell us how we wrong we are, and SGD to help us learn from our mistakes.

5. Now comes the fun part. We train our model. We feed it a bunch of images and let it try to guess what numbers they are, then we tell it how it did and let it adjust.

6. We finally test our model. We give it a new set of images it hasn't seen before, let it guess the numbers, and tally up how well it did. We print out the accuracy.

So basically, we've built a mini-AI that can look at a handwritten number and guess what it is!
'''

import torch
import torchvision
import torchvision.transforms as transforms

# Declare a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Download MNIST training dataset and apply transformations
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download MNIST test dataset and apply transformations
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define a feed-forward neural network
model = torch.nn.Sequential(
    torch.nn.Linear(28 * 28, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10),
)

# Define a Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training the model
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # flatten the inputs into 1D tensor
        inputs = inputs.view(inputs.shape[0], -1)

        # reset the parameter gradients
        optimizer.zero_grad()

        # forward pass, backward pass, and optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

# Testing the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        images = images.view(images.shape[0], -1)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))