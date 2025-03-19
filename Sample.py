import snntorch as snn
from snntorch import utils
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 128
data_path='MNIST'
num_classes = 10  # MNIST has 10 output classes

dtype = torch.float

transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)

print(f"The size of mnist_train is {len(mnist_train)}")

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)