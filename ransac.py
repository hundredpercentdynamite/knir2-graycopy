import numpy as np
from sklearn.neighbors import KDTree
import utils

def getMeanDistance(data):
    tree = KDTree(data, leaf_size=2)
    nearest_dist, nearest_ind = tree.query(data, k=8)
    mean = np.mean(nearest_dist[:, 1:])
    return mean

def calcDistanceToPlane(data, eq):
    a, b, c, d = eq
    distance = (a * data[:, 0] + b * data[:, 1] + c * data[:, 2] + d
                ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
    return distance

def ransac_plane(data, threshold=0.01, masterEq=None, iterations=1000):
    inliers = []
    i = 1
    equation = masterEq
    if masterEq is None:
        equation = [0, 0, 0, 0]
    finalPoints = []
    bestError = float('inf')
    zerosIdx = np.all(data == 0, axis=1)
    peaksIdx = np.all(data == 1, axis=1)
    fakeDist = threshold + 1,
    while i < iterations:
        planePoints = utils.getRandomPoints(data)
        normalVec = utils.getNormalVec(planePoints)
        if np.all(normalVec == 0.):
            continue
        a, b, c, d = utils.getPlaneParams(normalVec, planePoints)
        # расстояние от каждой точки до плоскости
        # distance = (a * data[:, 0] + b * data[:, 1] + c * data[:, 2] + d
        #             ) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        distance = calcDistanceToPlane(data, [a, b, c, d])
        distance[zerosIdx == True] = fakeDist
        distance[peaksIdx == True] = fakeDist
        idx_candidates = np.where(np.abs(distance) <= threshold)[0]
        error = float('inf')
        if (masterEq is not None):
            error = np.sum(calcDistanceToPlane(data[idx_candidates], masterEq))
            if (error <= bestError):
                equation = [a, b, c, d]
                inliers = idx_candidates
                finalPoints = planePoints
                bestError = error
        if len(idx_candidates) > len(inliers) and masterEq is None:
            equation = [a, b, c, d]
            inliers = idx_candidates
            finalPoints = planePoints
            bestError = error
        i += 1
    return equation, inliers, finalPoints

# def getClosestPointToZero(points):
#     min = math.inf
#     minIndex = -1
#     for i in range(len(points)):
#         norm = np.linalg.norm(points[i] - np.array([0, 0, 0]))
#         if norm < min:
#             min = norm
#             minIndex = i
#     return min, minIndex


def getPlane(flatData):
    mean = getMeanDistance(flatData)
    eq, idx_inliers, points = ransac_plane(flatData, mean)

    finalEq, finalIdxInliers, finalPoints = eq, idx_inliers, points

    return finalEq, finalIdxInliers, finalPoints

