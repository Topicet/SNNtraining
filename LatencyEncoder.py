import snntorch as snn
from Functions.function import rmse
from snntorch import utils, spikegen
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML
class LatencyEncoder():

    def __init__(self, batch_size=10, num_subsets=10):
        """
        Initializes the LatencyEncoder with specified batch size, number of classes, and subsets.
        
        Args:
            batch_size (int): Number of samples per batch.
            num_classes (int): Number of classes (digits) in the dataset.
            num_subsets (int): Number of subsets (images) to process.
        """
        self.batch_size = batch_size        
        self.num_subsets = num_subsets
        self.num_steps = 100  # Default number of time steps for latency encoding
        self.prepare_data()  # Prepare the dataset for encoding

    def prepare_data(self):
        """
        Prepares the MNIST dataset by selecting one sample for each digit (0-9) and loading it into a DataLoader.
        """
        # Define transformations for the MNIST dataset
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Resize images to 28x28
            transforms.Grayscale(),  # Convert images to grayscale
            transforms.ToTensor(),  # Convert images to tensors
            transforms.Normalize((0,), (1,))  # Normalize pixel values to [0, 1]
        ])

        # Load the full MNIST dataset
        full_dataset = datasets.MNIST('MNIST', train=True, download=True, transform=self.transform)

        indices = torch.randperm(len(full_dataset))[:self.num_subsets]
        self.dataset = torch.utils.data.Subset(full_dataset, indices)

        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.data = iter(self.train_loader)
        self.data_iterator, self.targets_iterator = next(self.data)
        self.targets_iterator = list(self.targets_iterator.numpy())

        print(f"Selected Digits in First Batch: {self.targets_iterator}")


    def spike_data(self, numberOfSteps, tau, threshold):
        """
        Encodes the dataset into spike trains using latency-based encoding.

        Args:
            numberOfSteps (int): Number of time steps for encoding.
            tau (float): Time constant for the encoding function.
            threshold (float): Threshold for spike generation.

        Returns:
            torch.Tensor: Latency-encoded spike trains.
        """
        return spikegen.latency(
            self.data_iterator, 
            num_steps=numberOfSteps, 
            tau=tau, 
            threshold=threshold, 
            normalize=True, 
            linear=True, 
            clip=True
        )

    def convert_to_time(self, data, tau=5, threshold=0.01):
        """
        Converts input data into spike times based on a given threshold and time constant.

        Args:
            data (torch.Tensor): Input data to convert.
            tau (float): Time constant for the conversion.
            threshold (float): Threshold for spike generation.

        Returns:
            torch.Tensor: Spike times for the input data.
        """
        # Compute spike times using the latency encoding formula
        spike_time = tau * torch.log(data / (data - threshold))
        return spike_time
    
    def save_all_visualizations(self, data):
        """
        Saves all visualizations including spiking animation, target number images, and raster plots.
        """
        self.animateSpiking(data)
        self.showTargetNumbers(data)
        self.showRasterPlot(data)

    def animate_spiking(self, data):
        """
        Creates and saves an animation of spiking activity for the first digit in the dataset.

        Args:
            data (torch.Tensor): The spiking data to animate.
        """
        data = data[:, 0, 0]  # Extract the first digit's spiking data
        fig, ax = plt.subplots()
        anim = splt.animator(data, fig, ax)  # Create the animation
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'  # Set ffmpeg path for saving
        HTML(anim.to_html5_video())  # Convert animation to HTML5 video
        anim.save("LatencyEncodingResults\\LatencyEncoded1stDigit.gif")  # Save animation as a GIF

    def show_target_numbers(self, data):
        """
        Visualizes and saves the average spiking activity for each digit in the dataset.

        Args:
            data (torch.Tensor): The spiking data to visualize.
        """
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # Create a grid of subplots for 10 digits
        
        for i, ax in enumerate(axes.flat):
            img = data[:, i].mean(dim=0).squeeze().cpu()  # Compute the average spiking activity for each digit
            ax.imshow(img, cmap='binary')  # Display the image in binary colormap
            ax.axis('off')  # Remove axes for cleaner visualization
            ax.set_title(f"Target Value: {self.targets_iterator[i]}")  # Set title with target digit

        plt.savefig("LatencyEncodingResults\\LatencyEncoded10Digits", bbox_inches="tight", dpi=300)  # Save the figure
        plt.close(fig)  # Close the figure to free memory

    def show_raster_plot(self, data):
        """
        Generates and saves a raster plot of spiking activity for the first digit in the dataset.

        Args:
            data (torch.Tensor): The spiking data to plot.
        """
        data = data[:, 0, 0]  # Extract the first digit's spiking data
        data = data.reshape((100, -1))  # Reshape data for raster plotting
        fig = plt.figure(facecolor="w", figsize=(10, 5))  # Create a figure
        ax = fig.add_subplot(111)
        splt.raster(data.view(self.num_steps, -1), ax, s=25, c="black")  # Generate raster plot

        plt.title("Input Layer")  # Set plot title
        plt.xlabel("Time step")  # Label x-axis
        plt.ylabel("Neuron Number")  # Label y-axis
        plt.savefig("LatencyEncodingResults\\LatencyEncoded1stDigitRasterPlot", bbox_inches="tight", dpi=300)  # Save plot
        plt.close(fig)  # Close the figure to free memory

    def dataset_summary(self, data):
        """
        Prints a summary of the latency-encoded dataset, including batch size, subsets, time steps,
        average firing numbers (AFN) per digit, and spike time distribution.

        Args:
            data (torch.Tensor): The spiking data to summarize.
        """
        print("Latency Dataset Information:")
        print(f"🔹 Batch Size: {self.batch_size}")  # Print batch size
        print(f"🔹 Number of Subsets (Total Images): {self.num_subsets}")  # Print number of subsets
        print(f"🔹 Time Steps: {self.num_steps}")  # Print number of time steps
        print(f"🔹 Data Shape: {data.shape}")  # Print shape of the spiking data

        # Compute average firing number (AFN) per digit
        afn_per_digit = data.sum(dim=(0, 2, 3, 4)) / self.num_steps
        print("Average Firing Number (AFN) per Digit:")
        for i, afn in enumerate(afn_per_digit):
            print(f"🔹 Digit {self.targets_iterator[i]}: {afn.item():.2f}")  # Print AFN for each digit

        # Print spike time distribution statistics
        print("Spike Time Distribution:")
        print(f"🔹 Min spike time: {data.min().item():.2f}")  # Print minimum spike time
        print(f"🔹 Max spike time: {data.max().item():.2f}")  # Print maximum spike time
        print(f"🔹 Mean spike time: {data.mean().item():.2f}")  # Print mean spike time