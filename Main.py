import RateEncoder, LatencyEncoder
from Functions.function import rmse, afr

if __name__ == "__main__":
    latencyEncoder = LatencyEncoder.LatencyEncoder()
    rateEncoder = RateEncoder.RateEncoder()


    latencyEncodedData = latencyEncoder.spike_data(numberOfSteps=100,tau=5,threshold=0.01)
    rateEncodedData = rateEncoder.spike_data(numberOfSteps=100, gain=1)

    print(len(latencyEncodedData))
    print(latencyEncodedData.shape)

    afn_rate = afr(rateEncodedData)
    afn_latency = afr(latencyEncodedData)

    print("AFN (Rate Coding):", afn_rate)
    print("AFN (Latency Coding):", afn_latency)

    latencyEncoder.showTargetNumber(latencyEncodedData)



