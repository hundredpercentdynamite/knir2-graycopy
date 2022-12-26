import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

annotationsDirectory = 'data/annotations'
directory = 'data/images'
width = 300.0
height = 205.0
standardPerspective = np.float32([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
imgSize = (int(width), int(height))

cv2.namedWindow("Image")


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


def getColorDistribution(image):
    blue, green, red = splitChannels(image * 255)
    figure = plt.figure(figsize=(15, 15), dpi=80)
    plot = figure.add_subplot(projection='3d')
    # copy_rgb.view_init(30, 300)
    # copy_rgb.view_init(15, 500)
    # plot.view_init(-5, 595)

    plot.set_title("RGB")
    plot.set_xlabel('Red')
    plot.set_ylabel('Green')
    plot.set_zlabel('Blue')

    plot.scatter(
        xs=red,
        ys=green,
        zs=blue,
        marker=".",
        s=1
    )

    plot.plot([0, 255], [0, 255], [0, 255], color="black", linestyle='-', linewidth=2)

    plt.show()
    return figure


def srgbToLinear(x):
    if x <= 0.0:
        return 0.0
    elif x >= 1.0:
        return 1.0
    elif x < 0.04045:
        return x / 12.92
    else:
        return pow((x + 0.055) / 1.055, 2.4)


def convertToLinearRgb(image):
    st = time.time()
    linearRgb = image

    for i in range(len(linearRgb)):
        for j in range(len(linearRgb[i])):
            for p in range(len(linearRgb[i][j])):
                value = linearRgb[i][j][p]
                newValue = srgbToLinear(value * (1 / 255.0))
                linearRgb[i][j][p] = newValue
    end = time.time()
    elapsed_time = end - st
    print('Execution time:', elapsed_time, 'seconds')
    return linearRgb


def getDocumentContour(imageMeta):
    contour = []
    for i in range(0, 4):
        x = imageMeta["regions"][0]["shape_attributes"]["all_points_x"][i]
        y = imageMeta["regions"][0]["shape_attributes"]["all_points_y"][i]
        contour.append([x, y])
    return np.float32(contour)


for docType in os.listdir(directory):
    docTypePath = os.path.join(directory, docType)
    docTypeAnnotation = os.path.join(annotationsDirectory, docType)
    for imgFolder in os.listdir(docTypePath):
        imgFolderPath = os.path.join(docTypePath, imgFolder)
        currentAnnotation = os.path.join(docTypeAnnotation, imgFolder + ".json")
        jsonAnnotation = None
        with open(currentAnnotation) as f:
            jsonAnnotation = json.load(f)
        metaData = jsonAnnotation['_via_img_metadata']
        metaKeys = metaData.keys()

        for imgName in os.listdir(imgFolderPath):
            imgPath = os.path.join(imgFolderPath, imgName)
            imgKey = next((key for key in metaKeys if imgName in key), None)
            if (imgKey):
                meta = metaData[imgKey]
                currentImage = cv2.imread(imgPath)
                documentContour = getDocumentContour(meta)
                transformMatrix = cv2.getPerspectiveTransform(documentContour, standardPerspective)

                transformedImage = cv2.warpPerspective(currentImage, transformMatrix, dsize=imgSize)
                linRgbImage = convertToLinearRgb(np.float32(transformedImage))
                cv2.imshow("Image", transformedImage)
                fig = getColorDistribution(linRgbImage)
                cv2.waitKey()

# TODO рефакторинг
