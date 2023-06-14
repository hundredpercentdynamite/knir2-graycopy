import pandas as pd
import json
import numpy as np
import image
import ransac
import plane
import utils
import proba
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import time
import os

dfName = 'banknotes.csv'
df = pd.read_csv(dfName, sep=';')

width = 300.0
height = 200.0


def getImages(images):
    fig = plt.figure(figsize=(10, 8))
    columns = 4
    rows = 2
    for i in range(1, columns * rows + 1):
        if i <= len(images):
            imgPath = images[i - 1]
            img = cv2.imread(imgPath)
            img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
            fig.add_subplot(rows, columns, i)
            plt.axis('off')
            imgPathSplitted = imgPath.split('/')
            plt.title(imgPathSplitted[-2] + '/' + imgPathSplitted[-1])
            plt.imshow(img)
    return fig


def predict(blackRatioThreshold=0.9, planeDelta=0.02, ransacIterations=1000):
    yYrue = []
    yScore = []
    yPred = []
    falseNegatives = []
    falsePositives = []
    errors = []
    for index, row in df.iterrows():
        st = time.time()

        imagePath = row['imagePath']
        p1 = np.array(row['p1'].split(',')).astype(np.float32)
        p2 = np.array(row['p2'].split(',')).astype(np.float32)
        p3 = np.array(row['p3'].split(',')).astype(np.float32)
        p4 = np.array(row['p4'].split(',')).astype(np.float32)
        contour = np.array([p1, p2, p3, p4])
        isCopy = int(row['isCopy'])

        linRgbImage = []
        try:
            linRgbImage = image.getLinearImageFromContourWithoutWarp(imagePath, contour, width, height)
        except Exception as e:
            errors.append(imagePath)
            continue
        flatImageData = utils.getFlatData(linRgbImage)

        eq, idx_inliers, __finalPoints = ransac.getPlane(flatImageData, ransacIterations)

        zerodFlat, zeroedIdx = plane.excludeMainPlane(flatImageData, eq, planeDelta)
        result, zeroRatio = proba.calcBlackRatio(flatImageData, zeroedIdx, blackRatioThreshold)
        yYrue.append(isCopy)
        yScore.append(result)
        yPred.append(zeroRatio)
        if isCopy == 1 and result == 0:
            falseNegatives.append(imagePath)
        if isCopy == 0 and result == 1:
            falsePositives.append(imagePath)
        end = time.time()
        elapsed_time = end - st
        print('Execution time: ', elapsed_time, 'seconds')
    return yYrue, yScore, yPred, falseNegatives, falsePositives, errors


def getScores(yT, yS, yP, beta=2):
    confusion = metrics.confusion_matrix(yT, yS)
    fScore = metrics.fbeta_score(yT, yS, beta=beta)
    aucScore = metrics.roc_auc_score(yT, yP)
    fpr, tpr, thresholds = metrics.roc_curve(yT, yP)
    return confusion, fScore, aucScore, fpr, tpr, thresholds


def getReport(confusion, fScore, aucScore, fpr, tpr, thresholds, fn, fp, errors, planeDelta, blackRatio, reportPath):
    confusionDisplay = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Не копия", "Копия"])
    confusionDisplay.plot()
    confusionDisplay.figure_.axes[0].set_xlabel('Предсказание')
    confusionDisplay.figure_.axes[0].set_ylabel('Истина')
    confusionDisplay.figure_.axes[0].set_title('Матрица ошибок')
    confusionDisplay.figure_.savefig(reportPath + '/confusion.png')

    rocDisplay = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucScore, estimator_name='Классификатор')
    rocDisplay.plot()
    rocDisplay.figure_.axes[0].set_xlabel('Доля неверных срабатываний (FPR)')
    rocDisplay.figure_.axes[0].set_ylabel('Чувствительность (TPR)')
    rocDisplay.figure_.axes[0].set_title('ROC-кривая')
    rocDisplay.figure_.savefig(reportPath + '/roc.png')

    if len(fn) > 0:
        falseNegativePlot = getImages(np.random.choice(np.array(fn), 8))
        falseNegativePlot.savefig(reportPath + '/falseNegatives.png')
    if len(fp) > 0:
        falsePositivePlot = getImages(np.random.choice(np.array(fp), 8))
        falsePositivePlot.savefig(reportPath + '/falsePositives.png')

    falseNegativeDf = pd.DataFrame({'path': fn})
    falsePositiveDf = pd.DataFrame({'path': fp})
    errorsDf = pd.DataFrame({'path': errors})

    falseNegativeDf.to_csv(reportPath + '/false_negative.csv')
    falsePositiveDf.to_csv(reportPath + '/false_positive.csv')
    errorsDf.to_csv(reportPath + '/errors.csv')

    scores = {
        'confusion': confusion.tolist(),
        'F-2 Score': fScore,
        'ROC-AUC SCORE': aucScore,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'blackThreshold': blackRatio,
        'planeDelta': planeDelta,
    }
    scores_json = json.dumps(scores, indent=4)
    with open(reportPath + '/score.json', "w") as outfile:
        outfile.write(scores_json)

    plt.show()

def saveToJson(reportPath, confusion, fScore, aucScore, fpr, tpr, thresholds, blackRatio, planeDelta):
    scores = {
        'confusion': confusion.tolist(),
        'F-2 Score': fScore,
        'ROC-AUC SCORE': aucScore,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'blackThreshold': blackRatio,
        'planeDelta': planeDelta,
    }
    scores_json = json.dumps(scores, indent=4)
    with open(reportPath + '/score.json', "w") as outfile:
        outfile.write(scores_json)


def calculate():
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    dataset = dfName.split('.')[0]
    reportPath = './report/' + dataset + '/' + dt_string
    os.mkdir(reportPath)

    blackRatio = 0.8
    planeDelta = 0.015
    ransacIters = 1000
    # yYrue, yScore, yPred, falseNegatives, falsePositives, errors = predict(blackRatio, planeDelta, ransacIters)
    yYrue, yScore, yPred, falseNegatives, falsePositives, errors = predict()
    confusion, fScore, aucScore, fpr, tpr, thresholds = getScores(yYrue, yScore, yPred, 2)
    getReport(confusion, fScore, aucScore, fpr, tpr, thresholds, falseNegatives, falsePositives, errors, planeDelta,
              blackRatio, reportPath)
    # for i in [0.001, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035]:
    #     planeDelta = i
    #     now = datetime.now()
    #     dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    #     reportPath = './report/' + dataset + '/ransaciters/' + dt_string
    #     os.mkdir(reportPath)
    #
    #     yYrue, yScore, yPred, falseNegatives, falsePositives, errors = predict(blackRatio, planeDelta, ransacIters)
    #     confusion, fScore, aucScore, fpr, tpr, thresholds = getScores(yYrue, yScore, yPred, 2)
    #     saveToJson(reportPath, confusion, fScore, aucScore, fpr, tpr, thresholds, blackRatio, planeDelta)


calculate()
