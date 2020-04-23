import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import time

from sklearn.metrics import *
from utils.constants import *

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def acc_plot():
    labels = ['32', '64', '128']
    a = np.array([87.393,89.409,89.667])
    b = np.array([87.352,88.839,90.299])

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, a, width, label='No Validation')
    rects2 = ax.bar(x + width/2, b, width, label='Validation')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acc. %')
    ax.set_xlabel('# of Timesteps')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim((85,91))

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.savefig("plots/acc.jpg")

def heatmap(y_true, y_pred, name):
    target_names = TARGET_NAMES[NUM_CLASSES]
    clf_cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, digits=4, target_names=target_names, output_dict=True)
    clf_report2 = classification_report(y_true, y_pred, digits=4,target_names=target_names)
    print(clf_report2)

    f, ax= plt.subplots(figsize=(6,5))
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :-3].T, 
                annot=True, cmap="Blues_r", fmt=".4f")
    ax.set_title("Classification Report")
    f.tight_layout()
    if os.path.isdir("plots/"+DATE):
        plt.savefig("plots/"+DATE+name+time.strftime('%H:%M:%S')+"_classification_report.jpg")
    else:
        os.mkdir("plots/"+DATE)
        plt.savefig("plots/"+DATE+name+time.strftime('%H:%M:%S')+"_classification_report.jpg")
    f.clf()

    f, ax= plt.subplots(figsize=(5,5))
    sns.heatmap(pd.DataFrame(clf_cm, index=target_names, columns=target_names).T, 
                    annot=True, cmap="BuGn", fmt="d")
    ax.set_title("Confusion Matrix")
    f.tight_layout()
    plt.savefig("plots/"+DATE+name+time.strftime('%H:%M:%S')+"_confusion_matrix.jpg")




