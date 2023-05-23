import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import image
import ransac
import diagram
import plane
import utils

annotationsDirectory = '../dlc-2021/cg/clips/annotations'
directory = '../dlc-2021/cg/clips/images'

width = 400.0
height = 315.0

cv2.namedWindow("Original")
cv2.namedWindow("Image")
cv2.namedWindow("WithoutPlaneImage")



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

                linRgbImage = image.getLinearImageWithoutPerspective(imgPath, meta)
                flatImageData = utils.getFlatData(linRgbImage) / 255
                eq, idx_inliers, finalPoints = ransac.getPlane(flatImageData)
                fig = diagram.getPlaneScatters(flatImageData, idx_inliers, finalPoints)
                projection, imageWithoutPlane, idx_cands = plane.getProjectionOnOrtoPlane(finalPoints, flatImageData, idx_inliers, eq)
                projectionFig = diagram.getProjectionPlaneScatter(projection, idx_cands)
                fig.show()
                fig2 = diagram.getPlaneScatters(imageWithoutPlane, idx_inliers, finalPoints)
                fig2.show()
                projectionFig.show()
                restoredImage = imageWithoutPlane.reshape(linRgbImage.shape)
                documentPixels = len(flatImageData[np.all(flatImageData > 0, axis=1)])
                zeroRatio = len(idx_cands) / documentPixels
                print("Black ratio", zeroRatio)
                cv2.imshow("Original", cv2.resize(cv2.imread(imgPath), (0, 0), fx=0.2, fy=0.2))
                cv2.imshow("Image", linRgbImage)
                cv2.imshow("WithoutPlaneImage", restoredImage)
                key = cv2.waitKey()
                break

# TODO рефакторинг
