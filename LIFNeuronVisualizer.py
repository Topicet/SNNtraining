import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

class LIFNeuronVisualizer:
    def __init__(self, R=5.1, C=5e-3, threshold=1, time_step=1e-3):
        self.R = R
        self.C = C
        self.threshold = threshold
        self.time_step = time_step
        self.tau_mem = R * C

    def leaky_integrate_and_fire(self, mem, cur):
        spk = mem > self.threshold
        mem = mem + (self.time_step / self.tau_mem) * (-mem + cur * self.R) - spk * self.threshold
        return mem, spk

    def plot_cur_mem_spk(self, cur, mem, spk, thr_line=False, vline=False, xlim=200, title="Hello", ylim_max2=1.25, save_path="./data"):
        fig, ax = plt.subplots(3, figsize=(8, 6), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1, 0.4]})

        # Input current
        ax[0].plot(cur, c="tab:orange")
        ax[0].set_ylim([0, 0.4])
        ax[0].set_xlim([0, xlim])
        ax[0].set_ylabel("Input Current ($I_{in}$)")
        if title:
            ax[0].set_title(title)

        # Membrane potential
        ax[1].plot(mem)
        ax[1].set_ylim([0, ylim_max2])
        ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
        if thr_line:
            ax[1].axhline(y=thr_line, alpha=0.25, linestyle="dashed", c="black", linewidth=2)
        ax[1].set_xlabel("Time step")

        # Output spikes
        import torch
        splt.raster(torch.tensor(spk, dtype=torch.float32), ax[2], s=400, c="black", marker="|")
        if vline:
            ax[2].axvline(x=vline, ymin=0, ymax=6.75, alpha=0.15, linestyle="dashed", c="black", linewidth=2, zorder=0, clip_on=False)
        ax[2].set_ylabel("Output spikes")
        ax[2].set_yticks([])

        if title:
            plt.savefig("x.jpg")
        else:
            plt.show()
