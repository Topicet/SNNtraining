import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch as snn
from RateEncoder import RateEncoder
from LatencyEncoder import LatencyEncoder
import os
import matplotlib.pyplot as plt

class SNNTrainer:
    def __init__(self, batch_size=10, num_steps=100, learning_rate=1e-3, epochs=20, encoder="rate", gain=1, num_substeps=1000, tau=5,threshold=0.005):
        """
        Initializes the SNNTrainer with training parameters and selects the encoding method.

        Args:
            batch_size (int): Number of samples per training batch.
            num_steps (int): Number of time steps in the spiking simulation.
            learning_rate (float): Learning rate for the Adam optimizer.
            epochs (int): Number of training epochs.
            encoder (str): Encoding method to use. Must be either "rate" or "latency".
            gain (float): Gain value used for rate encoding.
            num_substeps (int): Number of data samples to use from the dataset.
            tau (float): Time constant used in latency encoding.
            threshold (float): Threshold value for latency-based spike generation.

        Raises:
            SystemExit: If an invalid encoder type is provided.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        print(f"Using device: {self.device}")
        self.batch_size = batch_size
        self.num_steps = num_steps  # Number of time steps for spiking simulation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_loss_log = []  # Log to store training loss for each epoch

        self.num_substeps = num_substeps
        self.gain = gain
        self.encoding_type = encoder
        
        self.tau = tau
        self.threshold = threshold

        # Load encoded spike data and labels
        match encoder:
            case "rate":
                self.get_rate_encoded_data()
            case "latency":
                self.get_latency_encoded_data()
            case _:
                print("Invalid encoding type.")
                exit(0)

        self.labels = torch.tensor(self.encoder.targets_iterator).to(self.device)  # Target labels

        # Build the spiking neural network model
        self.model = self.SNNModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer

        self.print_config()


    def get_rate_encoded_data(self):
        self.encoder = RateEncoder(batch_size=self.batch_size, num_subsets=self.num_substeps)
        self.encoded_data = self.encoder.spike_data(numberOfSteps=self.num_steps, gain=self.gain)  # Encoded spike data

    def get_latency_encoded_data(self):
        self.encoder = LatencyEncoder(batch_size=self.batch_size, num_subsets=self.num_substeps)
        self.encoded_data = self.encoder.spike_data(numberOfSteps=self.num_steps,tau=self.tau, threshold=self.threshold) # Encoded spike data
    
    

    class SNNModel(nn.Module):
        def __init__(self):
            """
            Defines the architecture of the spiking neural network.
            """
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 100)  # Fully connected layer 1, Flatten image to 784, project to 100 neurons
            self.lif1 = snn.Leaky(beta=0.9)  # Leaky integrate-and-fire (LIF) neuron layer 1
            self.fc2 = nn.Linear(100, 10)  # Fully connected layer 2, Project to 10 output classes
            self.lif2 = snn.Leaky(beta=0.9)  # Leaky integrate-and-fire (LIF) neuron layer 2

        def forward(self, x, num_steps=100):
            """
            Forward pass through the SNN for a given number of time steps.
            Args:
                x: Input spike data (time, batch, features).
                num_steps: Number of time steps for simulation.
            Returns:
                Tensor of output spikes over time (time, batch, 10).
            """
            mem1 = self.lif1.init_leaky()  # Initialize membrane potential for LIF layer 1
            mem2 = self.lif2.init_leaky()  # Initialize membrane potential for LIF layer 2
            spk2_rec = []  # Record output spikes from the final layer

            for t in range(num_steps):
                # Process input through the first layer
                cur1 = self.fc1(x[t].view(x.size(1), -1))  # Flatten input
                spk1, mem1 = self.lif1(cur1, mem1)  # LIF neuron dynamics

                # Process through the second layer
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)  # LIF neuron dynamics
                spk2_rec.append(spk2)  # Record output spikes

            return torch.stack(spk2_rec)  # Stack spikes over time

    def train(self):
        """
        Trains the SNN model using mean squared error (MSE) loss.
        """
        self.model.train()  # Set model to training mode
        inputs = self.encoded_data.to(self.device).float()  # Input spike data
        targets = F.one_hot(self.labels, num_classes=10).float()  # One-hot encoded target labels

        for epoch in range(self.epochs):
            # Forward pass: compute output spikes
            output_spikes = self.model(inputs, self.num_steps)
            spike_sum = output_spikes.sum(dim=0)  # Sum spikes across time

            # Compute loss
            loss = F.mse_loss(spike_sum, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log training loss
            self.train_loss_log.append(loss.item())

            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.4f}")

        # Save the training loss plot
        self.save_loss_plot()
 

    def save_loss_plot(self):
        """
        Saves a detailed plot of the training loss over epochs.
        """
        base_filename = "TrainingResults/training_loss"
        extension = ".png"
        filename = base_filename + extension

        # Increment file version if filename already exists
        if os.path.exists(filename):
            while os.path.exists(filename):
                if "_" in filename:
                    base, digit = filename.rsplit("_", 1)
                    digit = digit.replace(extension, "")
                    if digit.isdigit():
                        filename = f"{base}_{int(digit) + 1}{extension}"
                    else:
                        filename = f"{base_filename}_1{extension}"
                else:
                    filename = f"{base_filename}_1{extension}"

        # Generate the plot
        epochs = list(range(1, len(self.train_loss_log) + 1))
        losses = self.train_loss_log
        min_loss = min(losses)
        min_epoch = losses.index(min_loss) + 1

        metadata = f"""Encoding: {self.encoding_type}
Batch Size: {self.batch_size}
LR: {self.learning_rate}
Epochs: {self.epochs}"""

        if self.encoding_type == "rate":
            metadata += f"\nGain: {self.gain}"
        else:
            metadata += f"\nTau: {self.tau}, Threshold: {self.threshold}"

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, losses, marker='o', label="Training Loss", color='blue')
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.legend()

        # Metadata box
        plt.gcf().text(1.02, 0.5, metadata, fontsize=10, va='center')

        # Min loss annotation
        plt.annotate(f"Min Loss: {min_loss:.4f}",
                    xy=(min_epoch, min_loss),
                    xytext=(min_epoch, min_loss + 0.01),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10,
                    color='black')

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the side text
        plt.savefig(filename, bbox_inches="tight")
        plt.close()



    def get_loss_log(self):
        """
        Returns the training loss log.
        """
        return self.train_loss_log
    

    def print_config(self):
        """
        Prints the key training parameters and model configuration.
        """
        print("\n--- SNN Trainer Configuration ---")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of steps: {self.num_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {self.epochs}")
        print(f"Input shape: {self.encoded_data.shape}")
        print(f"Labels shape: {self.labels.shape}")
        print("----------------------------------\n")
