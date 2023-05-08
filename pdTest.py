import pandas as pd
import json
import numpy as np
import image
import ransac
import plane
import utils
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import cv2
df = pd.read_csv('images-test.csv', sep=';')

width = 30.0
height = 20.0

def getImages(images):
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 2
    for i in range(0, columns * rows + 1):
        if i < len(images):
            imgPath = images[i]
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1)
            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            plt.title(imgPath.split('/')[-1])
            plt.imshow(img)
    return fig


def predict(blackRatioThreshold = 0.9):
    yYrue = []
    yScore = []
    yPred = []
    falseNegatives = []
    falsePositives = []
    for index, row in df.iterrows():
        imagePath = row['imagePath']
        p1 = np.array(row['p1'].split(',')).astype(np.float32)
        p2 = np.array(row['p2'].split(',')).astype(np.float32)
        p3 = np.array(row['p3'].split(',')).astype(np.float32)
        p4 = np.array(row['p4'].split(',')).astype(np.float32)
        contour = np.array([p1, p2, p3, p4])
        isCopy = int(row['isCopy'])

        linRgbImage = image.getLinearImageFromContour(imagePath, contour, width, height)
        flatImageData = utils.getFlatData(linRgbImage)

        eq, idx_inliers, __finalPoints = ransac.getPlane(flatImageData)

        zerodFlat, zeroedIdx = plane.excludeMainPlane(flatImageData, eq)
        # zeroedFlat, zeroedIdx = plane.excludeCandidates(flatImageData, idx_inliers)

        zeroRatio = len(zeroedIdx) / (width * height)
        result = 1 if zeroRatio > blackRatioThreshold else 0
        yYrue.append(isCopy)
        yScore.append(result)
        yPred.append(zeroRatio)
        if isCopy == 1 and result == 0:
            falseNegatives.append(imagePath)
        if isCopy == 0 and result == 1:
            falsePositives.append(imagePath)
    return yYrue, yScore, yPred, falseNegatives, falsePositives


yT, yS, yP, fn, fp = predict(0.9)


confusion = metrics.confusion_matrix(yT, yS)
print("Confusion", confusion)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Сopy", "Orig"])
disp.plot()
fScore = metrics.fbeta_score(yT, yS, beta=0.5)
print("F Score", fScore)
score = metrics.roc_auc_score(yT, yP)
print("ROC-AUC SCORE", score)
fpr, tpr, thresholds = metrics.roc_curve(yT, yP)
print("thresholds", thresholds)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=score, estimator_name='Classifier')
display.plot()

if len(fn) > 0:
    falseNegativePlot = getImages(np.random.choice(np.array(fn), 8))
if len(fp) > 0:
    falsePositivePlot = getImages(np.random.choice(np.array(fp), 8))




plt.show()
