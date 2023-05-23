import numpy as np

def calcBlackRatio(flatData, zeroed, threshold):
    documentPixels = len(flatData[np.all(flatData > 0, axis=1)])
    zeroRatio = len(zeroed) / documentPixels
    result = 1 if zeroRatio > threshold else 0
    return result, zeroRatio
