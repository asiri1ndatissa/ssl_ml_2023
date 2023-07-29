from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

with open('/Users/asiriindatissa/src/msc/ssl_ml_2023/tainFiles/sign_map.json', 'r') as f:
    sign_map = json.load(f)

def createConfusionMatrix(y_pred,y_true):

    # Extract the keys as a tuple to create the classes
    # classes = tuple(sign_map.keys())
    # print('classes', classes)

    classes = ('Good', 'Not Good', 'GE', 'GM', 'Go', 'Cant', 'Can', 'No', 'Bad', 'Dont Know', 'Know', 'Like',
                'Yes', 'Epa', 'Come', 'Yellow', 'Red', 'Orange', 'Green', 'Gold',
               'Brown', 'Blue')
    # classes = tuple(str(i) for i in np.unique(np.concatenate([y_pred, y_true])))
    # print('len(np.unique(y_pred))',len(np.unique(y_pred)), len(np.unique(y_true)) , len(np.unique(np.concatenate([y_pred, y_true]))))
    # if len(np.unique(y_pred)) == 23:
    #     classes = tuple(str(i) for i in range(0,23))
    # else:
    #     classes = tuple(str(i) for i in np.unique(np.concatenate([y_pred, y_true])))




    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
    #                      columns=[i for i in classes])
    # plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()

def classificationReport(y_true, y_pred,epoch):
    report = classification_report(y_true, y_pred)
    if epoch%10 == 0:
        print(report)

    # Now, create an image from the report
    # fig, ax = plt.subplots()
    # ax.text(0.5, 0.5, report, horizontalalignment='center', verticalalignment='center', fontsize=12)
    # plt.axis('off')

    # # Convert the plot to a PIL Image
    # buf = BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # img = Image.open(buf)

    # # Convert the PIL Image to a numpy array
    # img_arr = np.array(img)
    return report