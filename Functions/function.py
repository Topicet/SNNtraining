import numpy as np

def rmse(originalSignal, reconstuctedSignal):
    originalSignal = np.array(originalSignal)
    reconstuctedSignal = np.array(reconstuctedSignal)

    return np.sqrt(np.mean((originalSignal - reconstuctedSignal)**2))


def afr(spike_train):
    spike_train = np.array(spike_train)
    
    return np.sum(np.abs(spike_train)) / len(spike_train)