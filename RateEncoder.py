import snntorch as snn
from snntorch import utils, spikegen
import torch
from torch.utils.data import DataLoader
from Functions.function import rmse
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
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        full_dataset = datasets.MNIST('MNIST', train=True, download=True, transform=self.transform)

        #select digits 0-9
        digit_samples = {i: None for i in range(10)}
        for image, label in full_dataset:
            if digit_samples[label] is None:
                digit_samples[label] = (image, label)

            if all(digit_samples.values()): 
                break

        images, labels = zip(*digit_samples.values())
        self.dataset = list(zip(images, labels))

        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.data = iter(self.train_loader)
        self.data_iterator, self.targets_iterator = next(self.data)

        
        self.targets_iterator = list(self.targets_iterator.numpy())

        print(f"Selected Digits: {self.targets_iterator}")  # Should print [0,1,2,3,4,5,6,7,8,9]      

    def spike_data(self, numberOfSteps, gain):
        self.gain = gain
        return spikegen.rate(self.data_iterator, num_steps=numberOfSteps, gain=gain)
    
        
    def saveAllVisualizations(self, data):
        self.animateSpiking(data)
        self.showTargetNumbers(data)
        self.showRasterPlot(data)

    def animateSpiking(self, data):
        data = data[:, 0, 0]
        fig, ax = plt.subplots()
        anim = splt.animator(data, fig, ax)
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
        HTML(anim.to_html5_video())
        anim.save("RateEncodingResults\\RateEncoded1stDigit.gif")

    def showTargetNumbers(self, data):        
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
        plt.savefig("RateEncodingResults\\RateEncoded1stDigitRasterPlot", bbox_inches="tight", dpi=300)
        plt.close(fig) 

    def dataset_summary(self, data):
        print("Rate Dataset Information:")
        print(f"🔹 Batch Size: {self.batch_size}")
        print(f"🔹 Number of Subsets (Total Images): {self.num_subsets}")
        print(f"🔹 Time Steps: {self.num_steps}")
        print(f"🔹 Data Shape: {data.shape}")

        afn_per_digit = data.sum(dim=(0, 2, 3, 4)) / self.num_steps
        print("Average Firing Number (AFN) per Digit:")
        for i, afn in enumerate(afn_per_digit):
            print(f"🔹 Digit {self.targets_iterator[i]}: {afn.item():.2f}")

        print("Spike Time Distribution:")
        print(f"🔹 Min spike time: {data.min().item():.2f}")
        print(f"🔹 Max spike time: {data.max().item():.2f}")
        print(f"🔹 Mean spike time: {data.mean().item():.2f}")

    def reconstruct_images(self, data, filename="RateEncodingResults/RateEncoded_Reconstruction.png"):
        reconstructed_images = data.mean(dim=0).squeeze().cpu()

        original_images = self.data_iterator.squeeze().cpu()


        fig, axes = plt.subplots(2, 10, figsize=(15, 3))

        for i in range(10):

            print(f'RMSE for figure {i}: {rmse(original_images[i], reconstructed_images[i])}')

            axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].set_title(f"Original {self.targets_iterator[i]}")
            axes[0, i].axis('off')

            axes[1, i].imshow(reconstructed_images[i], cmap='gray')
            axes[1, i].set_title(f"New {self.targets_iterator[i]}")
            axes[1, i].axis('off')

        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"✅ Reconstructed images saved as {filename}")

        #return reconstructed_images  # Returning reconstructed images for further analysis