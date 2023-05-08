import numpy as np
import random

def splitChannels(image):
    blue = []
    green = []
    red = []
    for i in range(len(image)):
        for b, g, r in image[i]:
            blue.append(b)
            green.append(g)
            red.append(r)
    return blue, green, red


def getFlatData(data):
    width = len(data[0])
    height = len(data)
    length = width * height
    return data.reshape(length, 3)


def getNormalVec(points):
    if (len(points) != 3): raise Exception("Length of points should be equal 3")
    vecA = points[1] - points[0]
    vecB = points[2] - points[0]
    normal = np.cross(vecA, vecB)
    return normal

def getRandomPoints(data, amount = 3):
    idx_samples = random.sample(range(len(data)), amount)
    return data[idx_samples]


def getPlaneParams(normal, points):
    if (len(points) != 3): raise Exception("Length of points should be equal 3")
    # 𝑎𝑥+𝑏𝑦+𝑐𝑧+𝑑=0
    normalLength = np.linalg.norm(normal)
    a, b, c = normal / normalLength
    # 𝑑=-(𝑎𝑥+𝑏𝑦+𝑐𝑧)
    d = -np.sum(normal * points[1])
    return a, b, c, d
