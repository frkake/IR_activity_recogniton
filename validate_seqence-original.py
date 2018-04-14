"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import ResearchModels
from data import DataSet
import os
import pandas as pd
import numpy as np
import cv2
import glob
import csv
from keras.utils import np_utils
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score

classes = sorted(os.listdir("/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/train"))


def get_class_one_hot(class_str):
    """Given a class as a string, return its number in the classes
    list. This lets us encode and one-hot it for training."""
    # Encode it first.
    label_encoded = classes.index(class_str)

    # Now one-hot it.
    label_hot = np_utils.to_categorical(label_encoded, len(classes))
    label_hot = label_hot[0]  # just get a single row

    return label_hot


def getseq_csv(framecsv, labelcsv):
    frames = np.loadtxt(framecsv, dtype=np.float32, delimiter=',').reshape(-1, 16, 16, 1)
    df = pd.read_csv(labelcsv)

    i = 0
    x = []
    _y = []
    start = df.ix[:, 0][0]
    for i in range(len(df)):
        framenum = df.ix[:, 0][i]
        seqdata = frames[framenum - 9:framenum + 1]
        label = df.ix[:, 1][i]
        x.append(seqdata)
        _y.append(get_class_one_hot(label))

    return start, np.array(x), np.array(_y)


def validate(data_type, model, seq_length=10, saved_model=None,
             concat=False, class_limit=None, image_shape=None, framecsv=False, scoredcsv=False):
    batch_size = 16

    modelname = saved_model.split('/')[-1].split('-')[0]
    dataname = scoredcsv.split('/')[-1]

    save_directory = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/prediction/" + \
                     str(modelname) + "-" + str(dataname)
    try:
        os.mkdir(save_directory)
    except FileExistsError:
        pass

    # Get the data and process it.
    start, datas, labels = getseq_csv(framecsv, scoredcsv)

    # val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(classes), model, seq_length, saved_model)

    filenumber = 0
    nb_samples = 0
    nb_incorrect = 0
    textback = np.zeros((320, 800), dtype=np.uint8)

    pred = []
    ans = []
    for i, (data, label) in enumerate(zip(datas, labels)):
        # textback.fill(0)
        data = data.reshape(1, -1, 16, 16, 1)
        # nowdata = data[0][0]

        resultpred = rm.model.predict(
            x=data,
            batch_size=5,
            verbose=0
        )

        resultclass = np.argmax(resultpred)

        label = np.argmax(label)

        predictclass = classes[int(resultclass)]
        answerclass = classes[label]
        pred.append(predictclass)
        ans.append(answerclass)

    # if classes[int(resultclass)] != classes[label]:
    #         print()
    #         print('Row num:', start + i)
    #         print('Answer: {}'.format(classes[label]))
    #         print('Predict: {}'.format(classes[int(resultclass)]))
    #
    #         nb_incorrect += 1
    #     nb_samples += 1
    # print('The number of incorrect is {0} in {1} samples'.format(nb_incorrect, nb_samples))
    # print('Accuracy is ', 100 * (1 - float(nb_incorrect) / nb_samples), '%')

    return pred, ans



    # print(classes)


def main():
    modelpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/learnedmodels/"
    framepath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/sequences/Seqs2/"
    scoredpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/scored/"
    confpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/confusion_matrixies/"

    saved_models = glob.glob(modelpath + '/*.hdf5')
    framecsvs = glob.glob(framepath + '/tempval*.csv')
    # scoredcsvs = glob.glob(scoredpath + '/*.csv')

    summary = []
    merge = np.zeros(shape=(18, 18))
    for saved_model in saved_models:
        model = saved_model.split('/')[-1].split('-')[0]
        print('model: ', model)
        preds, anses = [], []

        for framecsv in framecsvs:
            framecsvname = framecsv.split('/')[-1]
            scoredcsv = glob.glob(scoredpath + model + '-original-' + framecsvname + '/*.csv')[0]

            if model == 'mlp' or model == 'lstm':
                data_type = 'features'
                image_shape = None

            else:
                data_type = 'images'
                image_shape = (16, 16, 1)
                load_to_memory = False

            if model == 'mlp':
                concat = True
            else:
                concat = False

            pred, ans = validate(data_type, model, saved_model=saved_model,
                                 concat=concat, image_shape=image_shape, framecsv=framecsv, scoredcsv=scoredcsv)
            preds.extend(pred)
            anses.extend(ans)
            print(framecsvname)

        confmat = confusion_matrix(anses, preds, labels=classes)
        # repo = classification_report(anses, preds)
        acc = accuracy_score(anses, preds)
        precision, recall, fscore, support = score(anses, preds)
        print(model)
        print(confmat)
        print(acc)
        print(precision)
        print(fscore)
        print(type(precision))
        summary.append([model, acc])
        precision = np.array(precision)
        recall = np.array(recall)
        fscore = np.array(fscore)

        merge += confmat

        csvarray = pd.DataFrame(confmat, index=classes, columns=classes)
        csvarray.to_csv(confpath + model + '-original-' + str(acc) + '.csv', index=True, header=True, sep=',')

        csvprecision = pd.DataFrame(precision, index=classes, columns=["precision"])
        csvprecision.to_csv(confpath + model + '-original-precision.csv', index=True, header=True, sep=',')
        csvrecall = pd.DataFrame(recall, index=classes, columns=["recall"])
        csvrecall.to_csv(confpath + model + '-original-recall.csv', index=True, header=True, sep=',')
        csvfscore = pd.DataFrame(fscore, index=classes, columns=["fscore"])
        csvfscore.to_csv(confpath + model + '-original-fscore.csv', index=True, header=True, sep=',')

    csvarray = pd.DataFrame(merge, index=classes, columns=classes)
    csvarray.to_csv(confpath + 'original-merge_matrix.csv', index=True, header=True, sep=',')

    for sm in summary:
        with open(
                '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/confusion_matrixies/summary-original.csv',
                'a') as smr:
            writer = csv.writer(smr, lineterminator='\n')
            writer.writerow(sm)


if __name__ == '__main__':
    main()
