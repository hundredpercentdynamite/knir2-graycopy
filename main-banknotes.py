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


annotationsDirectory = './data/annotations'
directory = './data/images'

width = 400.0
height = 315.0

cv2.namedWindow("Original")
cv2.namedWindow("Image")
cv2.namedWindow("WithoutPlaneImage")



keyboardKey = ''


def onPress(event):
    global keyboardKey
    print(event.key)
    keyboardKey = event.key
    if keyboardKey == " " or keyboardKey == "q":
        plt.close('all')


for imgFolder in os.listdir(directory):
    imgFolderPath = os.path.join(directory, imgFolder)
    currentAnnotation = os.path.join(annotationsDirectory, imgFolder + ".json")
    jsonAnnotation = None
    with open(currentAnnotation) as f:
        jsonAnnotation = json.load(f)
    metaKeys = jsonAnnotation.keys()

    for imgName in os.listdir(imgFolderPath):
        if keyboardKey == ord('q'):
            print('breaked!')
            keyboardKey = ''
            break
        imgPath = os.path.join(imgFolderPath, imgName)
        imgKey = next((key for key in metaKeys if imgName in key), None)
        print("imgPath", imgPath)
        if (imgKey):
            meta = jsonAnnotation[imgKey]

            linRgbImage = image.getLinearImageWithoutPerspective(imgPath, meta)
            flatImageData = utils.getFlatData(linRgbImage) / 255
            eq, idx_inliers, finalPoints = ransac.getPlane(flatImageData)
            fig = diagram.getPlaneScatters(flatImageData, idx_inliers, finalPoints)
            projection, imageWithoutPlane, idx_cands = plane.getProjectionOnOrtoPlane(finalPoints, flatImageData, idx_inliers, eq)
            # projectionFig = diagram.getProjectionPlaneScatter(projection, idx_cands)
            fig.show()
            # fig2 = diagram.getPlaneScatters(imageWithoutPlane, idx_inliers, finalPoints)
            # fig2.show()
            # projectionFig.show()
            restoredImage = imageWithoutPlane.reshape(linRgbImage.shape)
            documentPixels = len(flatImageData[np.all(flatImageData > 0, axis=1)])
            zeroRatio = len(idx_cands) / documentPixels
            print("Black ratio", zeroRatio)
            # cv2.imshow("Original", cv2.resize(cv2.imread(imgPath), (0, 0), fx=0.2, fy=0.2))
            cv2.imshow("Image", linRgbImage)
            cv2.imshow("WithoutPlaneImage", restoredImage)
            keyboardKey = cv2.waitKey()
            # break

# TODO рефакторинг
