import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import image
import ransac
import diagram
import plane
import utils

annotationsDirectory = '../dlc-2021/cg/clips/annotations'
directory = '../dlc-2021/cg/clips/images'

# width = 150.0
width = 300.0
# height = 100.0
height = 215.0

# cv2.namedWindow("Original")
cv2.namedWindow("Image")
cv2.namedWindow("WithoutPlaneImage")


# cv2.namedWindow("Hist")



key = ''


def onPress(event):
    global key
    print(event.key)
    key = event.key
    if key == " " or key == "q":
        plt.close('all')


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
        if key == ord('q'):
            print('breaked!')
            key = ''
            break

        for imgName in os.listdir(imgFolderPath):
            imgPath = os.path.join(imgFolderPath, imgName)
            imgKey = next((key for key in metaKeys if imgName in key), None)
            print("imgPath", imgPath)
            if (imgKey):
                meta = metaData[imgKey]
                linRgbImage = image.getLinearImage(imgPath, meta, width, height)
                flatImageData = utils.getFlatData(linRgbImage) / 255
                eq, idx_inliers, finalPoints = ransac.getPlane(flatImageData)
                fig = diagram.getPlaneScatters(flatImageData, idx_inliers, finalPoints)
                projection, imageWithoutPlane, idx_cands = plane.getProjectionOnOrtoPlane(finalPoints, flatImageData, idx_inliers, eq)
                projectionFig = diagram.getProjectionPlaneScatter(projection, idx_cands)
                fig.show()
                projectionFig.show()
                restoredImage = imageWithoutPlane.reshape(int(height), int(width), 3)
                zeroRatio = len(idx_cands) / (width * height)
                print("Black ratio", zeroRatio)
                # cv2.imshow("Original", cv2.imread(imgPath))
                cv2.imshow("Image", linRgbImage)
                cv2.imshow("WithoutPlaneImage", restoredImage)
                key = cv2.waitKey()
                break

# TODO рефакторинг
