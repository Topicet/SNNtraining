import RateEncoder, LatencyEncoder
from Functions.function import rmse, afr

if __name__ == "__main__":
    latencyEncoder = LatencyEncoder.LatencyEncoder()
    rateEncoder = RateEncoder.RateEncoder()


    latencyEncodedData = latencyEncoder.spike_data(numberOfSteps=100,tau=5,threshold=0.01)
    rateEncodedData = rateEncoder.spike_data(numberOfSteps=100, gain=1)

    latencyEncoder.saveAllVisualizations(latencyEncodedData)
    latencyEncoder.dataset_summary(latencyEncodedData)
    #rateEncoder.saveAllVisualizations(rateEncodedData)



