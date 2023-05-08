import os
import json
import image
import pandas as pd
import numpy as np

lowcolorAnnotationsDirectory = '../dlc-2021/lowcolor/clips/annotations'
lowcolorDirectory = '../dlc-2021/lowcolor/clips/images'

cgAnnotationsDirectory = '../dlc-2021/cg/clips/annotations'
cgDirectory = '../dlc-2021/cg/clips/images'


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
                    [p1, p2, p3, p4] = image.getDocumentContour(meta)

                    data['id'].append(imgKey)
                    data['imagePath'].append(imgPath)
                    data['p1'].append(','.join(np.char.mod('%f', p1)))
                    data['p2'].append(','.join(np.char.mod('%f', p2)))
                    data['p3'].append(','.join(np.char.mod('%f', p3)))
                    data['p4'].append(','.join(np.char.mod('%f', p4)))
                    data['isCopy'].append(1 if isCopy else 0)
                break
    df = pd.DataFrame.from_dict(data)
    return df


dfCopy = makeDataFrame(cgDirectory, cgAnnotationsDirectory, True)
dfLowcolor = makeDataFrame(lowcolorDirectory, lowcolorAnnotationsDirectory, False)

frames = [dfCopy, dfLowcolor]

dfAll = pd.concat(frames)

dfAll.to_csv('images-single.csv', index=False, sep=';')

print(dfAll)
