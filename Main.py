import RateEncoder, LatencyEncoder
from LIFNeuronVisualizer import LIFNeuronVisualizer
from Functions.function import rmse, afr
import numpy as np
import torch
import matplotlib.pyplot as plt
from LIFNeuronVisualizer import LIFNeuronVisualizer
from TrainSNN import SNNTrainer


if __name__ == "__main__":   

    my_snn_trainer = SNNTrainer(batch_size=20,num_steps=100,learning_rate=1e-3,epochs=150)
    my_snn_trainer.train()


    # latencyEncoder = LatencyEncoder.LatencyEncoder()
    # rateEncoder = RateEncoder.RateEncoder()
    # latencyEncodedData = latencyEncoder.spike_data(numberOfSteps=100,tau=5,threshold=0.005)
    # #rateEncodedData = rateEncoder.spike_data(numberOfSteps=100, gain=0.33)
    # latencyEncoder.reconstruct_images(latencyEncodedData)
    # latencyEncoder.saveAllVisualizations(latencyEncodedData)
    # latencyEncoder.dataset_summary(latencyEncodedData)
    #rateEncoder.dataset_summary(rateEncodedData)
    #rateEncoder.reconstruct_images(rateEncodedData)
    #rateEncoder.saveAllVisualizations(rateEncodedData)
