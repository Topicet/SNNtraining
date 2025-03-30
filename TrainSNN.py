import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import snntorch as snn
from RateEncoder import RateEncoder
import matplotlib.pyplot as plt

class SNNTrainer:
    def __init__(self, batch_size=10, num_steps=100, learning_rate=1e-3, epochs=20):
        """
        Initializes the SNNTrainer class with training parameters and model setup.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        print(f"Using device: {self.device}")
        self.batch_size = batch_size
        self.num_steps = num_steps  # Number of time steps for spiking simulation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_loss_log = []  # Log to store training loss for each epoch

        # Load encoded spike data and labels
        self.encoder = RateEncoder(batch_size=batch_size)
        self.spike_data = self.encoder.spike_data(numberOfSteps=num_steps, gain=0.33)  # Encoded spike data
        self.labels = torch.tensor(self.encoder.targets_iterator).to(self.device)  # Target labels

        # Build the spiking neural network model
        self.model = self.SNNModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer

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
        inputs = self.spike_data.to(self.device).float()  # Input spike data
        targets = F.one_hot(self.labels, num_classes=10).float()  # One-hot encoded target labels

        printCounter = 0
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
            if printCounter % 2 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.4f}")
                printCounter += 1

        # Save the training loss plot
        self.save_loss_plot()

    def save_loss_plot(self):
        """
        Saves a plot of the training loss over epochs.
        """
        plt.plot(self.train_loss_log)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.savefig("training_loss.png")  # Save plot as an image
        plt.close()

    def get_loss_log(self):
        """
        Returns the training loss log.
        """
        return self.train_loss_log