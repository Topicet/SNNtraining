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

    def __init__(self, batch_size=10, num_subsets=1000):
        self.batch_size = batch_size        
        self.num_subsets = num_subsets
        self.num_steps = 100
        self.prepare_data()


    def prepare_data(self):
        """
        Prepares a dataset of N random MNIST digits.
        Loads the dataset, applies transformations, and samples a subset of size `num_subsets`.
        """
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))
        ])

        # Load the full MNIST dataset
        full_dataset = datasets.MNIST('MNIST', train=True, download=True, transform=self.transform)

        # Randomly select `num_subsets` samples
        indices = torch.randperm(len(full_dataset))[:self.num_subsets]
        self.dataset = torch.utils.data.Subset(full_dataset, indices)

        # Create DataLoader
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Load first batch for inspection
        self.data = iter(self.train_loader)
        self.data_iterator, self.targets_iterator = next(self.data)
        self.targets_iterator = list(self.targets_iterator.numpy())

        print(f"Selected Digits in First Batch: {self.targets_iterator}")


    def spike_data(self, numberOfSteps, gain):
        """
        Encodes the data into spike trains using rate-based encoding.
        Args:
            numberOfSteps (int): Number of time steps for spike encoding.
            gain (float): Gain factor to scale the input data for spike generation.
        Returns:
            torch.Tensor: Spike-encoded data with shape (time_steps, batch_size, channels, height, width).
        """
        self.gain = gain  # Store the gain factor as an attribute
        # Generate spike trains using rate-based encoding
        return spikegen.rate(self.data_iterator, num_steps=numberOfSteps, gain=gain)
    
        
    def save_all_visualizations(self, data):
        self.animateSpiking(data)
        self.showTargetNumbers(data)
        self.showRasterPlot(data)

    def animate_spiking(self, data):
        """
        Creates and saves an animation of the spiking activity for the first digit.
        Args:
            data (torch.Tensor): Spike-encoded data.
        """
        data = data[:, 0, 0]  # Extract spiking data for the first digit
        fig, ax = plt.subplots()
        anim = splt.animator(data, fig, ax)  # Generate animation
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'  # Set ffmpeg path for saving
        HTML(anim.to_html5_video())  # Convert animation to HTML5 video
        anim.save("RateEncodingResults\\RateEncoded1stDigit.gif")  # Save animation as a GIF

    def show_target_numbers(self, data):
        """
        Visualizes the average spiking activity for each digit and saves the result.
        Args:
            data (torch.Tensor): Spike-encoded data.
        """
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))  # Create a 2x5 grid for visualization
        
        for i, ax in enumerate(axes.flat):
            img = data[:, i].mean(dim=0).squeeze().cpu()  # Compute average spiking activity for each digit
            ax.imshow(img, cmap='binary')  # Display the image
            ax.axis('off')  # Remove axis for cleaner visualization
            ax.set_title(f"Target Value: {self.targets_iterator[i]}")  # Add title with target digit

        plt.savefig("RateEncodingResults\\RateEncoded10Digits", bbox_inches="tight", dpi=300)  # Save the plot
        plt.close(fig)  # Close the figure to free memory

    def show_raster_plot(self, data):
        """
        Generates and saves a raster plot of the spiking activity for the first digit.
        Args:
            data (torch.Tensor): Spike-encoded data.
        """
        data = data[:, 0, 0]  # Extract spiking data for the first digit
        data = data.reshape((100, -1))  # Reshape data for raster plot
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(data, ax, s=1.5, c="black")  # Generate raster plot

        plt.title("Input Layer")  # Add title
        plt.xlabel("Time step")  # Label x-axis
        plt.ylabel("Neuron Number")  # Label y-axis
        plt.savefig("RateEncodingResults\\RateEncoded1stDigitRasterPlot", bbox_inches="tight", dpi=300)  # Save plot
        plt.close(fig)  # Close the figure to free memory

    def dataset_summary(self, data):
        """
        Prints a summary of the dataset, including batch size, subsets, time steps, and average firing numbers.
        Args:
            data (torch.Tensor): Spike-encoded data.
        """
        print("Rate Dataset Information:")
        print(f"🔹 Batch Size: {self.batch_size}")
        print(f"🔹 Number of Subsets (Total Images): {self.num_subsets}")
        print(f"🔹 Time Steps: {self.num_steps}")
        print(f"🔹 Data Shape: {data.shape}")

        # Calculate average firing number (AFN) per digit
        afn_per_digit = data.sum(dim=(0, 2, 3, 4)) / self.num_steps
        print("Average Firing Number (AFN) per Digit:")
        for i, afn in enumerate(afn_per_digit):
            print(f"🔹 Digit {self.targets_iterator[i]}: {afn.item():.2f}")

        # Print spike time distribution statistics
        print("Spike Time Distribution:")
        print(f"🔹 Min spike time: {data.min().item():.2f}")
        print(f"🔹 Max spike time: {data.max().item():.2f}")
        print(f"🔹 Mean spike time: {data.mean().item():.2f}")

    def reconstruct_images(self, data, filename="RateEncodingResults/RateEncoded_Reconstruction.png"):
        """
        Reconstructs images from the spike-encoded data, compares them with the original images, 
        and saves the results.
        Args:
            data (torch.Tensor): Spike-encoded data.
            filename (str): Path to save the reconstructed images.
        """
        reconstructed_images = data.mean(dim=0).squeeze().cpu()  # Compute reconstructed images
        original_images = self.data_iterator.squeeze().cpu()  # Extract original images

        fig, axes = plt.subplots(2, 10, figsize=(15, 3))  # Create a 2x10 grid for visualization

        for i in range(10):
            # Compute RMSE between original and reconstructed images
            print(f'RMSE for figure {i}: {rmse(original_images[i], reconstructed_images[i])}')

            # Display original image
            axes[0, i].imshow(original_images[i], cmap='gray')
            axes[0, i].set_title(f"Original {self.targets_iterator[i]}")
            axes[0, i].axis('off')

            # Display reconstructed image
            axes[1, i].imshow(reconstructed_images[i], cmap='gray')
            axes[1, i].set_title(f"New {self.targets_iterator[i]}")
            axes[1, i].axis('off')

        plt.savefig(filename, bbox_inches="tight", dpi=300)  # Save the plot
        plt.close(fig)  # Close the figure to free memory
        print(f"✅ Reconstructed images saved as {filename}")

        #return reconstructed_images  # Returning reconstructed images for further analysis