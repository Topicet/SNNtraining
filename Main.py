import RateEncoder, LatencyEncoder
from LIFNeuronVisualizer import LIFNeuronVisualizer
from Functions.function import rmse, afr
import numpy as np
import torch
import matplotlib.pyplot as plt
from LIFNeuronVisualizer import LIFNeuronVisualizer  # Ensure correct import

def run_lif_experiments():
    time_steps = 300
    input_current = np.zeros(time_steps)

    input_current[0] = 0.3
    input_current[25:] += 0.3

    hyperparams = [
        {"R": 5.1, "C": 5e-3, "threshold": 1.0, "label": "Baseline"},
        {"R": 10, "C": 5e-3, "threshold": 1.0, "label": "Higher Resistance"},
        {"R": 5.1, "C": 1e-2, "threshold": 1.0, "label": "Higher Capacitance"},
        {"R": 5.1, "C": 5e-3, "threshold": 0.8, "label": "Lower Threshold"},
        {"R": 5.1, "C": 5e-3, "threshold": 1.2, "label": "Higher Threshold"},
    ]

    fig, axs = plt.subplots(len(hyperparams), 3, figsize=(10, 2 * len(hyperparams)), sharex=True)

    for idx, params in enumerate(hyperparams):
        lif = LIFNeuronVisualizer(R=params["R"], C=params["C"], threshold=params["threshold"])

        mem = 0
        mem_trace = []
        spk_trace = []

        for t in range(time_steps):
            mem, spk = lif.leaky_integrate_and_fire(mem, input_current[t])
            mem_trace.append(mem)
            spk_trace.append(spk)

        spk_tensor = torch.tensor(spk_trace, dtype=torch.float32)

        axs[idx, 0].plot(input_current, c="tab:orange")
        axs[idx, 0].set_ylabel(f"{params['label']}\n$I_{{in}}$")
        axs[idx, 0].set_ylim([0, 0.4])

        axs[idx, 1].plot(mem_trace, c="tab:blue")
        axs[idx, 1].axhline(y=params["threshold"], linestyle="dashed", color="black", alpha=0.5)
        axs[idx, 1].set_ylabel("$U_{mem}$")
        axs[idx, 1].set_ylim([0, 1.5])

        from snntorch import spikeplot as splt 
        splt.raster(spk_tensor, axs[idx, 2], s=100, c="black", marker="|")
        axs[idx, 2].set_ylabel("Spikes")
        axs[idx, 2].set_yticks([])

    axs[-1, 0].set_xlabel("Time Step")
    axs[-1, 1].set_xlabel("Time Step")
    axs[-1, 2].set_xlabel("Time Step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_lif_experiments()
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

       




