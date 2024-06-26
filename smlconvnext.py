import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# Define transformation for the input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Load custom dataset using ImageFolder
full_dataset = datasets.ImageFolder(root='\archive', transform=transform)

# Split dataset into training and testing subsets (80-20 split)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Define data loaders for training and testing
trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
testloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Check if GPU is available, otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load pre-trained ConvNet_Tiny model and move it to GPU
model = models.convnext_tiny(weights='IMAGENET1K_V1').to(device)
num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_ftrs, len(train_dataset.dataset.classes)).to(device)  # Adjust output size according to your dataset

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Train the model
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)  # Move inputs and labels to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Print loss after every 10 epochs
    if (epoch + 1) % 10 == 0:
        print('[Epoch %d] loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        running_loss = 0.0

print('Finished Training')

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)  # Move inputs and labels to GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test set: %d %%' % (100 * correct / total))
