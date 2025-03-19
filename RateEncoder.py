import snntorch as snn
from snntorch import utils, spikegen
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

class RateEncoder():

    def __init__(self, batch_size=10, num_classes=10, num_subsets=10):
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
        #print(f"The size of mnist_train is {len(self.dataset)}")
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        self.data = iter(self.train_loader)
        self.data_iterator, self.targets_iterator = next(self.data)
      

    def spike_data(self, numberOfSteps, gain):
        self.gain = gain
        return spikegen.rate(self.data_iterator, num_steps=numberOfSteps, gain=gain)

    def animateSpiking(self, data):
        data = data[:, 0, 0]
        fig, ax = plt.subplots()
        anim = splt.animator(data, fig, ax)
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        HTML(anim.to_html5_video())
        print(f"The corresponding target is: {self.targets_iterator[0]}")
        plt.show()    

    def showTargetNumber(self, data):        
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))  
        
        for i, ax in enumerate(axes.flat):
            img = data[:, i].mean(dim=0).squeeze().cpu()
            ax.imshow(img, cmap='binary')
            ax.axis('off')
            ax.set_title(f"Target Value: {self.targets_iterator[i]}")

        plt.savefig("RateEncodingResults\\RateEncoded10Digits", bbox_inches="tight", dpi=300)
        plt.close(fig) 

    def showRasterPlot(self, data):
        data = data[:, 0, 0]
        data = data.reshape((100, -1))
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(data, ax, s=1.5, c="black")

        plt.title("Input Layer")
        plt.xlabel("Time step")
        plt.ylabel("Neuron Number")
        plt.show()