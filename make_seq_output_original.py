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
from keras.utils import np_utils
from scipy.stats import rankdata

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
             concat=False, class_limit=None, image_shape=None, framecsv=False):
    batch_size = 16

    modelname = saved_model.split('/')[-1].split('-')[0]
    dataname = framecsv.split('/')[-1]

    save_directory = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/prediction/" + \
                     str(modelname) + "-original" + "-" + str(dataname)
    try:
        os.mkdir(save_directory)
    except FileExistsError:
        pass

    # Get the data and process it.
    start, datas = getseq_csv(framecsv)

    # val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(classes), model, seq_length, saved_model)

    filenumber = 0
    textback = np.zeros((320, 800), dtype=np.uint8)

    for i, data in enumerate(datas):
        textback.fill(0)
        data = data.reshape(1, -1, 16, 16, 1)
        nowdata = data[0][9]

        resultpred = rm.model.predict(
            x=data,
            batch_size=5,
            verbose=0
        )

        resultclass = np.argmax(resultpred)

        # resultclass is classes[int(resultclass)]
        tempmax = np.max(nowdata)
        tempmin = np.min(nowdata)
        pixval = (nowdata - tempmin) * 255 / (tempmax - tempmin)

        pixval = np.asarray(pixval, dtype=np.uint8)

        pixcel = cv2.resize(pixval, None, fx=20, fy=20, interpolation=cv2.INTER_AREA)
        pixcel = cv2.flip(pixcel, 1)
        cv2.putText(textback, classes[int(resultclass)], (25, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,
                    (255, 255, 255), 6)
        cv2.putText(textback, "frame: " + str(start + i), (600, 270), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                    (255, 255, 255), 1)
        showfig = cv2.hconcat([pixcel, textback])

        cv2.imwrite(save_directory + "/" + dataname + "-original" + "-{0:03d}.jpg".format(start + i), showfig)

        with open(save_directory + "/" + dataname + "-original" + ".csv", 'a') as fresult:
            writer = csv.writer(fresult, lineterminator='\n')
            writer.writerow([start + i, classes[int(resultclass)]])

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

        cv2.imwrite(save_directory + "/" + dataname + "-original" + "-{0:03d}.jpg".format(i), showfig)


def main():

    modelpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/learnedmodels/"
    framepath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/sequences/Seqs/"

    saved_models = glob.glob(modelpath+'/*.hdf5')
    framecsvs = glob.glob(framepath + '/tempval*.csv')

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
                     concat=concat, image_shape=image_shape, framecsv=framecsv)


    # saved_model = ""
    # framecsv =""
    # if model == 'mlp' or model == 'lstm':
    #     data_type = 'features'
    #     image_shape = None
    #
    # else:
    #     data_type = 'images'
    #     image_shape = (16, 16, 1)
    #     load_to_memory = False
    #
    # if model == 'mlp':
    #     concat = True
    # else:
    #     concat = False
    #
    # validate(data_type, model, saved_model=saved_model,
    #          concat=concat, image_shape=image_shape, framecsv=framecsv)



if __name__ == '__main__':
    main()
