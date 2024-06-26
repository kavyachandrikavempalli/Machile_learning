import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

dataset = ImageFolder(root='/home/kvempall/archive', transform=transform)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

type(data_loader)

# for images, labels in data_loader:
#     # Images will be batch_size x 3 x 224 x 224 (assuming 3-channel RGB images)
#     # Visualize your augmented images
#     for image in images:
#         plt.imshow(image.permute(1, 2, 0))  # Convert from tensor to numpy array and rearrange channels
#         plt.show()