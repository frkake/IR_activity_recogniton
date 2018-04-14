import matplotlib.pyplot as plt
import pandas as pd
import os

conf_path = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/confusion_matrixies/"
model = "crnn"
mode = "original"

TRAIN_PATH = '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/train'
classes = sorted(os.listdir(TRAIN_PATH))
left = range(len(classes))

precision = pd.read_csv(conf_path + model + "-" + mode + "-precision.csv")
recall = pd.read_csv(conf_path + model + "-" + mode + "-recall.csv")
fscore = pd.read_csv(conf_path + model + "-" + mode + "-fscore.csv")

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(221)
ax1.bar(left, precision['precision'], tick_label=classes, color='red')

ax2 = fig.add_subplot(222)
ax2.bar(left, recall['recall'], label=classes, color='blue')

ax3 = fig.add_subplot(223)
ax3.bar(left, fscore['fscore'], label=classes, color='black')

ax1.set_xlabel('Class', fontsize=15)
ax1.set_ylabel('Precision', fontsize=15)
ax1.set_ylim([0, 1])
ax1.grid()
ax2.set_xlabel('Class', fontsize=15)
ax2.set_ylabel('Recall', fontsize=15)
ax2.set_ylim([0, 1])
ax2.grid()
ax3.set_ylabel('F_measure', fontsize=15)
ax3.set_ylim([0, 1])
ax3.grid()

plt.savefig(conf_path + model + "-" + mode + "-summary.jpg")
plt.show()
