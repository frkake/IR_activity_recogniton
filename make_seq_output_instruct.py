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
import csv
import glob
from sklearn.externals import joblib
from keras.utils import np_utils
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

classes = sorted(os.listdir("/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/train"))


def classindex(classname):
    return classes.index(classname)


def get_class_one_hot(class_str):
    """Given a class as a string, return its number in the classes
    list. This lets us encode and one-hot it for training."""
    # Encode it first.
    label_encoded = classes.index(class_str)

    # Now one-hot it.
    label_hot = np_utils.to_categorical(label_encoded, len(classes))
    label_hot = label_hot[0]  # just get a single row

    return label_hot


def getseq_csv(framecsv):
    frames = np.loadtxt(framecsv, dtype=np.float32, delimiter=',').reshape(-1, 16, 16, 1)

    i = 0
    x = []
    start = 10
    for i in range(len(frames)):
        if i == 0:
            continue
        framenum = i + 9
        if framenum >= len(frames):
            break
        seqdata = frames[framenum - 9:framenum + 1]
        x.append(seqdata)

    return start, np.array(x)


def validate(data_type, model, seq_length=10, saved_model=None,
             concat=False, class_limit=None, image_shape=None, framecsv=False, knn_model=None):
    nextbias = 0.4
    back1bias = 0.5
    back2bias = 0.6
    batch_size = 16

    modelname = saved_model.split('/')[-1].split('-')[0]
    dataname = framecsv.split('/')[-1]

    save_directory = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/prediction/" + \
                     str(modelname) + "-instruct" + "-" + str(dataname)
    try:
        os.mkdir(save_directory)
    except FileExistsError:
        pass

    # Get the data and process it.
    start, datas = getseq_csv(framecsv)

    # val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(classes), model, seq_length, saved_model)
    knn = joblib.load(knn_model)

    filenumber = 0
    textback = np.zeros((320, 800), dtype=np.uint8)

    for i, data in enumerate(datas):
        nextbias = 0.3
        back1bias = 0.5
        back2bias = 0.6
        cls_flag = 0
        batch_size = 16

        textback.fill(0)
        data = data.reshape(1, -1, 16, 16, 1)

        nowdata = data[0][9]

        resultpred = rm.model.predict(
            x=data,
            batch_size=5,
            verbose=0
        )

        resultclass = np.argmax(resultpred)

        if i == 0:
            nowstatus = classes[resultclass]
            continue

        prestatus = nowstatus
        preindex = classes.index(prestatus)

        if prestatus == 'Nobody':
            candilist = ['Nobody', 'AreaEntry']
            nextlist = ['Walk']
            back1list = ['AreaExit']
            back2list = ['Walk']

        elif prestatus == 'AreaEntry':
            candilist = ['AreaEntry', 'Walk']
            nextlist = ['AreaExit', 'StandingFloor', 'FallFloor']
            back1list = ['Nobody']
            back2list = ['AreaExit']

        elif prestatus == 'Walk':
            candilist = ['Walk', 'AreaExit', 'StandingFloor', 'FallFloor']
            nextlist = ['Nobody', 'LayingFloor']
            back1list = ['AreaEntry', 'StandingFloor', 'Sit']
            back2list = ['Nobody', 'StandFloor', 'Stand']

        elif prestatus == 'AreaExit':
            candilist = ['AreaExit', 'Nobody']
            nextlist = ['AreaEntry']
            back1list = ['Walk', 'FallFloor', 'StandingFloor']
            back2list = ['AreaEntry', 'StandingFloor']

        elif prestatus == 'StandingFloor':
            candilist = ['StandingFloor', 'Sit', 'FallFloor', 'Walk']
            nextlist = ['SittingBedrail', 'LayingFloor', 'FallFloor']
            back1list = ['Walk', 'StandFloor', 'Stand', 'AreaExit']
            back2list = ['AreaEntry', 'LayingFloor', 'SittingBedrail']

        elif prestatus == 'Sit':
            candilist = ['Sit', 'SittingBedrail']
            nextlist = ['BedEntry', 'Stand']
            back1list = ['StandingFloor', 'AreaEntry', 'FallFloor', 'Walk']
            back2list = ['Walk', 'StandFloor', 'AreaExit']

        elif prestatus == 'SittingBedrail':
            candilist = ['SittingBedrail', 'BedEntry', 'Stand']
            nextlist = ['SittingBed', 'StandingFloor']
            back1list = ['Sit', 'BedExit']
            back2list = ['StandingFloor', 'SittingBed']

        elif prestatus == 'BedEntry':
            candilist = ['BedEntry', 'SittingBed']
            nextlist = ['Down', 'BedExit']
            back1list = ['SittingBedrail', 'Stand']
            back2list = ['Sit', 'BedExit']

        elif prestatus == 'SittingBed':
            candilist = ['SittingBed', 'Down', 'BedExit']
            nextlist = ['LayingBed', 'SittingBedrail']
            back1list = ['BedEntry', 'WakeUp']
            back2list = ['SittingBedrail', 'LayingBed']

        elif prestatus == 'Down':
            candilist = ['Down', 'LayingBed']
            nextlist = ['FallBed', 'WakeUp']
            back1list = ['SittingBed', 'BedExit']
            back2list = ['BedEntry', 'WakeUp']

        elif prestatus == 'LayingBed':
            candilist = ['LayingBed', 'WakeUp', 'FallBed']
            nextlist = ['SittingBed', 'LayingFloor']
            back1list = ['Down']
            back2list = ['SittingBed']

        elif prestatus == 'WakeUp':
            candilist = ['WakeUp', 'SittingBed']
            nextlist = ['Down', 'BedExit']
            back1list = ['LayingBed', 'FallBed']
            back2list = ['Down']

        elif prestatus == 'BedExit':
            candilist = ['BedExit', 'SittingBedrail']
            nextlist = ['BedEntry', 'Stand']
            back1list = ['SittingBed', 'Down']
            back2list = ['WakeUp', 'BedEntry']

        elif prestatus == 'Stand':
            candilist = ['Stand', 'StandingFloor']
            nextlist = ['Walk', 'FallFloor', 'Sit']
            back1list = ['SittingBedrail', 'BedEntry']
            back2list = ['BedExit', 'Sit']

        elif prestatus == 'FallBed':
            candilist = ['FallBed', 'LayingFloor']
            nextlist = ['StandFloor']
            back1list = ['LayingBed', 'WakeUp']
            back2list = ['Down']

        elif prestatus == 'LayingFloor':
            candilist = ['LayingFloor', 'StandFloor']
            nextlist = ['StandingFloor']
            back1list = ['FallBed']
            back2list = ['LayingBed']

        elif prestatus == 'StandFloor':
            candilist = ['StandFloor', 'StandingFloor']
            nextlist = ['Walk', 'FallFloor', 'Sit']
            back1list = ['LayingFloor']
            back2list = ['FallBed']

        elif prestatus == 'FallFloor':
            candilist = ['FallFloor', 'LayingFloor']
            nextlist = ['StandFloor']
            back1list = ['Walk', 'StandingFloor', 'AreaExit', 'Sit']
            back2list = ['AreaEntry', 'StandingFloor', 'Walk', 'StandFloor', 'Stand']

        else:
            print('Error!')

        candipro = []
        for j, candiname in enumerate(candilist):
            candipro.append(resultpred[0][classindex(candiname)])

        for k, nextname in enumerate(nextlist):
            candipro.append(resultpred[0][classindex(nextname)] - nextbias)

        for l, back1name in enumerate(back1list):
            candipro.append(resultpred[0][classindex(back1name)] - back1bias)

        for m, back2name in enumerate(back2list):
            candipro.append(resultpred[0][classindex(back2name)] - back2bias)

        candilist.extend(nextlist)
        candilist.extend(back1list)
        candilist.extend(back2list)

        # print(candilist)
        topclass = candilist[candipro.index(max(candipro))]

        if topclass == 'AreaEntry':
            data = data.reshape(-1, 256)

            maxtemp9 = np.argmax(data[9])
            maxtemp8 = np.argmax(data[8])
            maxtemp0 = np.argmax(data[0])
            sumtemp = np.sum(data[9])
            ny, nx = np.divmod(maxtemp9, 16)
            py0, px0 = np.divmod(maxtemp0, 16)
            py1, px1 = np.divmod(maxtemp8, 16)
            mx1 = nx - px1
            my1 = ny - py1
            mx10 = nx - px0
            my10 = ny - py0

            x = np.array([data[9][maxtemp9], sumtemp, maxtemp9, mx1, my1, mx10, my10])

            kr = knn.predict([x])

            if kr == 1:
                topclass = 'Nobody'

        # resultclass is classes[int(resultclass)]
        tempmax = np.max(nowdata)
        tempmin = np.min(nowdata)
        pixval = (nowdata - tempmin) * 255 / (tempmax - tempmin)

        pixval = np.asarray(pixval, dtype=np.uint8)

        pixcel = cv2.resize(pixval, None, fx=20, fy=20, interpolation=cv2.INTER_AREA)
        pixcel = cv2.flip(pixcel, 1)
        cv2.putText(textback, topclass, (25, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                    (255, 255, 255), 6)
        cv2.putText(textback, "frame: " + str(start + i), (600, 270), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 255), 1)
        showfig = cv2.hconcat([pixcel, textback])

        cv2.imwrite(save_directory + "/" + dataname + "-instruct" + "-{0:03d}.jpg".format(start + i), showfig)
        nowstatus = topclass

        with open(save_directory + "/" + dataname + "-instruct" + ".csv", 'a') as fresult:
            writer = csv.writer(fresult, lineterminator='\n')
            writer.writerow([start + i, topclass])

    for i, data in zip(range(start), datas):
        textback.fill(0)
        data = data.reshape(1, -1, 16, 16, 1)
        data = datas[i]
        nowdata = data[0]

        # resultclass is classes[int(resultclass)]
        tempmax = np.max(nowdata)
        tempmin = np.min(nowdata)
        pixval = (nowdata - tempmin) * 255 / (tempmax - tempmin)

        pixval = np.asarray(pixval, dtype=np.uint8)

        pixcel = cv2.resize(pixval, None, fx=20, fy=20, interpolation=cv2.INTER_AREA)
        pixcel = cv2.flip(pixcel, 1)

        cv2.putText(textback, "frame: " + str(i), (600, 270), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 255), 1)
        showfig = cv2.hconcat([pixcel, textback])

        cv2.imwrite(save_directory + "/" + dataname + "-instruct" + "-{0:03d}.jpg".format(i), showfig)


def main():
    modelpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/learnedmodels/"
    framepath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/sequences/Seqs/"
    knnmodel = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/KNN5-NAe-.sav"

    saved_models = sorted(glob.glob(modelpath + '/*.hdf5'))
    framecsvs = sorted(glob.glob(framepath + '/tempval*.csv'))

    for saved_model in saved_models:
        model = saved_model.split('/')[-1].split('-')[0]

        print('model: ', model)
        for framecsv in framecsvs:
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

            validate(data_type, model, saved_model=saved_model,
                     concat=concat, image_shape=image_shape, framecsv=framecsv, knn_model=knnmodel)


if __name__ == '__main__':
    main()
