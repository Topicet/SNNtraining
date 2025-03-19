import snntorch as snn
from snntorch import utils, spikegen
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML



class Encoder():

    def __init__(self, batch_size=128, num_classes=10, num_subsets=10):
        self.batch_size = batch_size        
        self.num_classes = num_classes
        self.num_subsets = num_subsets
        self.num_steps = 100
        self.prepare_data()


    def prepare_data(self):
        self.transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])
        
        self.dataset = datasets.MNIST('MNIST', train=True, download=True, transform=self.transform)
        self.dataset = utils.data_subset(self.dataset, self.num_subsets)
        print(f"The size of mnist_train is {len(self.dataset)}")
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.data = iter(self.train_loader)
        self.data_iterator, self.targets_iterator = next(self.data)
      

    def spike_data(self, numberOfSteps, gain):
        self.gain = gain
        data = spikegen.rate(self.data_iterator, num_steps=numberOfSteps, gain=gain)
        return data[:, 0, 0]

    def animateSpiking(self, data):
        fig, ax = plt.subplots()
        anim = splt.animator(data, fig, ax)
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        HTML(anim.to_html5_video())
        print(f"The corresponding target is: {self.targets_iterator[0]}")
        plt.show()    

    def showTargetNumber(self, data):
        plt.figure(facecolor="w")
        plt.subplot(1,2,1)
        plt.imshow(data.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
        plt.axis('off')
        plt.title(f'Gain = {self.gain}')
        plt.show()

    def showRasterPlot(self, data):
        data = data.reshape((100, -1))
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(data, ax, s=1.5, c="black")

        plt.title("Input Layer")
        plt.xlabel("Time step")
        plt.ylabel("Neuron Number")
        plt.show()

    


if __name__ == "__main__":
    encoder = Encoder()
    myData = encoder.spike_data(numberOfSteps=100,gain=1)

    encoder.animateSpiking(myData)
    encoder.showTargetNumber(myData)
    encoder.showRasterPlot(myData)
