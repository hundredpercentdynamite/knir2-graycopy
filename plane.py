import numpy as np
import utils

def excludeMainPlane(flatData, equation, threshold=0.02):
    a, b, c, d = equation
    # расстояние от каждой точки до плоскости
    zerosIdx = np.all(flatData == 0, axis=1)
    peaksIdx = np.all(flatData == 1, axis=1)
    fakeDist = threshold + 1,

    distance = (a * flatData[:, 0] + b * flatData[:, 1] + c * flatData[:, 2] + d
                ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    distance[zerosIdx == True] = fakeDist
    distance[peaksIdx == True] = fakeDist
    idx_candidates = np.where(np.abs(distance) <= threshold)[0]
    zeroedFlat = np.copy(flatData)
    zeroedFlat[idx_candidates] = np.array([0, 0, 0])

    return zeroedFlat, idx_candidates

def excludeCandidates(flatData, inliers):
    zeroedFlat = np.copy(flatData)
    zeroedFlat[inliers] = np.array([0, 0, 0])

    return zeroedFlat, inliers


def getProjectionOnOrtoPlane(points, flatData, inliers, equation):
    baseVec = np.array([1, 1, 1])
    normalVec = utils.getNormalVec(points)
    ortVec = np.cross(normalVec, baseVec)

    left = [ortVec, normalVec]
    withoutMainPlane, idx_candidates = excludeMainPlane(flatData, equation)
    # withoutMainPlane, idx_candidates = excludeCandidates(flatData, inliers)

    projection = []
    for point in flatData:
        projection.append(np.matmul(left, point))
    return np.array(projection), withoutMainPlane, idx_candidates
