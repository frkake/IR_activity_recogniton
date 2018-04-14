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
from scipy.stats import rankdata


def rescale_list(input_list, size):
    """Given a list and a size, return a rescaled/samples list. For example,
    if we want a list of size 5 and we have a list of size 25, return a new
    list of size five which is every 5th element of the origina list."""
    print(len(input_list))
    # assert len(input_list) >= size

    # Get the number to skip between iterations.
    # skip = len(input_list) // size

    # Build our new output.
    # output = [input_list[i] for i in range(0, len(input_list), skip)]
    output = input_list[:size]
    print(output.shape)
    # Cut off the last one if needed.
    # print(output)

    return output


def validate(data_type, model, seq_length=10, saved_model=None,
             concat=False, class_limit=None, image_shape=None):
    batch_size = 16

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # val_generator = data.frame_generator(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    # results = rm.model.evaluate_generator(
    #     generator=val_generator,
    #     val_samples=17)

    flnames = []
    data_csv = pd.read_csv(
        '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/data_file.csv')

    flnames = list(data_csv[data_csv.iloc[:, 0] == 'test'].iloc[:, 2])
    fcategories = list(data_csv[data_csv.iloc[:, 0] == 'test'].iloc[:, 1])

    filenumber = 0
    nb_samples = 0
    nb_incorrect = 0
    for filename in flnames:
        fnumber = flnames[filenumber]
        fcategory = fcategories[filenumber]
        filenumber += 1
        allpath_f = '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/test/' + fcategory + '/' + fnumber + '.csv'

        if not os.path.isfile(allpath_f):
            continue

        f = np.loadtxt(allpath_f, delimiter=',').reshape((-1, 16, 16, 1))

        if len(f) > seq_length:
            skip = len(f) // seq_length
            f = f[::skip]
            f = f[:seq_length]
        else:
            continue

        f = f.reshape((1, -1, 16, 16, 1))

        resultpred = rm.model.predict(
            x=f,
            batch_size=5,
            verbose=0
        )

        resultclass = np.argmax(resultpred)

        # print(np.squeeze(resultpred).shape)
        # print(resultclass.shape)

        if data.classes[int(resultclass)] != fcategory:
            print(filename)
            print('Answer: {}'.format(fcategory))
            print('Predict: {}'.format(data.classes[int(resultclass)]))
            # print('classes: {}'.format(data.classes))

            rankindex = np.argsort(np.squeeze(resultpred))[::-1]
            rankclass = []
            for i in rankindex:
                rankclass.append(data.classes[i])

            print('Predict_possibility: {}\n'.format(rankclass))
            nb_incorrect += 1
        nb_samples += 1
    print('The number of incorrect is {0} in {1} samples'.format(nb_incorrect, nb_samples))

    print(data.classes)

def main():
    model = 'crnn'
    saved_model = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/checkpoints/crnn-images.960-0.097.hdf5"

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
             concat=concat, image_shape=image_shape)


if __name__ == '__main__':
    main()
