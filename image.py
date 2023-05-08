import cv2
import numpy as np
import time



def getDocumentContour(imageMeta, padding=30):
    contour = []
    for i in range(0, 4):
        x = imageMeta["regions"][0]["shape_attributes"]["all_points_x"][i]
        y = imageMeta["regions"][0]["shape_attributes"]["all_points_y"][i]
        contour.append([x, y])
    return np.float32(contour)


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
    print('Execution time stupid:', elapsed_time, 'seconds')
    return linearRgb

# LUT
table = []
for i in range(256):
    value = i / 255
    x = None
    if value < 0.04045:
        x = value / 12.92
    else:
        x = pow((value + 0.055) / 1.055, 2.4)
    table.append(x * 255)
table = np.array(table, np.uint8)

def gammaCorrection(src):
    # st = time.time()
    image = cv2.LUT(src, table)
    # end = time.time()
    # elapsed_time = end - st
    # print('Execution time LUT:', elapsed_time, 'seconds')
    return image


def getLinearImage(imgPath, annotation, width, height):
    standardPerspective = np.float32([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    imgSize = (int(width), int(height))

    currentImage = cv2.imread(imgPath)
    documentContour = getDocumentContour(annotation)
    transformMatrix = cv2.getPerspectiveTransform(documentContour, standardPerspective)

    transformedImage = cv2.warpPerspective(currentImage, transformMatrix, dsize=imgSize)
    # linRgbImage = convertToLinearRgb(np.float32(transformedImage))
    linRgbImage = gammaCorrection(np.uint8(transformedImage))
    linRgbImage = cv2.GaussianBlur(linRgbImage, (9, 9), 0)
    return linRgbImage


def getLinearImageFromContour(imgPath, contour, width, height):
    standardPerspective = np.float32([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    imgSize = (int(width), int(height))

    currentImage = cv2.imread(imgPath)
    documentContour = contour
    transformMatrix = cv2.getPerspectiveTransform(documentContour, standardPerspective)

    transformedImage = cv2.warpPerspective(currentImage, transformMatrix, dsize=imgSize)
    linRgbImage = gammaCorrection(np.uint8(transformedImage))
    linRgbImage = cv2.GaussianBlur(linRgbImage, (9, 9), 0)
    return linRgbImage / 255
