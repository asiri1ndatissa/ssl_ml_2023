from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

with open('/Users/asiriindatissa/src/msc/ssl_ml_2023/tainFiles/sign_map.json', 'r') as f:
    sign_map = json.load(f)

def createConfusionMatrix(y_pred,y_true):

    classes = ('Good', 'Not Good', 'GE', 'GM', 'Go', 'Cant', 'Can', 'No', 'Bad', 'Dont Know', 'Know', 'Like',
                'Yes', 'Epa', 'Come', 'Yellow', 'Red', 'Orange', 'Green', 'Gold',
               'Brown', 'Blue')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    return sn.heatmap(df_cm, annot=True).get_figure()

def classificationReport(y_true, y_pred,epoch):
    report = classification_report(y_true, y_pred)
    if epoch%10 == 0:
        print(report)
    return report