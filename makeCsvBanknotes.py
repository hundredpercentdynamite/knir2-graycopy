import os
import json
import image
import pandas as pd
import numpy as np

# lowcolorAnnotationsDirectory = './data/annotations'
# lowcolorDirectory = '../dlc-2021/lowcolor/clips/images'

cgAnnotationsDirectory = './data/annotations'
cgDirectory = './data/images'

def makeDataFrame(directory, annotationsDirectory, isCopy):
    data = {
        'id': [],
        'imagePath': [],
        'p1': [],
        'p2': [],
        'p3': [],
        'p4': [],
        'isCopy': [],
    }

    for imgFolder in os.listdir(cgDirectory):
        imgFolderPath = os.path.join(cgDirectory, imgFolder)
        currentAnnotation = os.path.join(cgAnnotationsDirectory, imgFolder + ".json")
        jsonAnnotation = None
        with open(currentAnnotation) as f:
            jsonAnnotation = json.load(f)
        metaKeys = jsonAnnotation.keys()

        for imgName in os.listdir(imgFolderPath):
            imgPath = os.path.join(imgFolderPath, imgName)
            imgKey = next((key for key in metaKeys if imgName in key), None)
            if (imgKey):
                meta = jsonAnnotation[imgKey]
                [p1, p2, p3, p4] = image.getDocumentContour(meta)

                data['id'].append(imgKey)
                data['imagePath'].append(imgPath)
                data['p1'].append(','.join(np.char.mod('%f', p1)))
                data['p2'].append(','.join(np.char.mod('%f', p2)))
                data['p3'].append(','.join(np.char.mod('%f', p3)))
                data['p4'].append(','.join(np.char.mod('%f', p4)))
                data['isCopy'].append(1 if isCopy else 0)
    df = pd.DataFrame.from_dict(data)
    return df


dfCopy = makeDataFrame(cgDirectory, cgAnnotationsDirectory, True)
# dfLowcolor = makeDataFrame(lowcolorDirectory, lowcolorAnnotationsDirectory, False)

# frames = [dfCopy, dfLowcolor]

# dfAll = pd.concat(frames)

dfCopy.to_csv('banknotes.csv', index=False, sep=';')

# print(dfAll)
