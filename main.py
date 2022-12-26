import os
import json
import cv2
import numpy as np

annotationsDirectory = 'data/annotations'
directory = 'data/images'
width = 720.0
height = 493.0
standardPerspective = np.float32([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
imgSize = (int(width), int(height))

cv2.namedWindow("Image")

def getBanknoteContour(meta):
    contour = []
    for i in range(0, 4):
        x = meta["regions"][0]["shape_attributes"]["all_points_x"][i]
        y = meta["regions"][0]["shape_attributes"]["all_points_y"][i]
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
        print(metaKeys)
        for imgName in os.listdir(imgFolderPath):
            imgPath = os.path.join(imgFolderPath, imgName)
            imgKey = next((key for key in metaKeys if imgName in key), None)
            if (imgKey):
                meta = metaData[imgKey]
                currentImage = cv2.imread(imgPath)
                contour = getBanknoteContour(meta)
                transformMatrix = cv2.getPerspectiveTransform(contour, standardPerspective)

                transformedImage = cv2.warpPerspective(currentImage, transformMatrix, dsize=imgSize)
                cv2.imshow("Image", transformedImage)

                cv2.waitKey()



