import os
import json
import numpy as np

dir = "./report/images-single/blackratioiter"
print("blackThreshold | Accuracy | Precision | Recall | FPR | F2-score")

results = []
for iteration in os.listdir(dir):
    scorePath = dir + "/" + iteration + "/score.json"
    scoreData = None
    with open(scorePath) as f:
        scoreData = json.load(f)
    blackThreshold = scoreData["blackThreshold"]
    f2Score = scoreData["F-2 Score"]
    [tn, fp], [fn, tp] = np.array(scoreData["confusion"])
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = (tp) / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    results.append([round(blackThreshold, 3), round(accuracy, 3), round(precision, 3), round(recall, 3), round(fpr, 3), round(f2Score, 3)])

sorted = np.sort(results, axis=0)
print(sorted)
