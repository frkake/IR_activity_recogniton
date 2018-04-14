"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import pandas as pd
import sys
import operator
import cv2
from processor import process_image
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

TRAIN_PATH = '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/train'
TEST_PATH = '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/test'

traindir_list = sorted(os.listdir(TRAIN_PATH))
testdir_list = sorted(os.listdir(TEST_PATH))


class DataSet():
    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = './data/sequences/'
        self.max_frames = 300  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()
        self.data2 = self.get_data2()

        # Get the classes.
        self.classes = self.get_classes()
        self.classes2 = self.get_classes2()

        # Now do some minor data cleaning.
        self.data = self.clean_data()
        self.data2 = self.clean_data2()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open('./data/data_file.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    @staticmethod
    def get_data2():
        """Load our data from file."""
        with open('./data/data_file2.csv', 'r') as fin2:
            reader2 = csv.reader(fin2)
            data2 = list(reader2)

        return data2

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def clean_data2(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean2 = []
        for item in self.data2:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes2:
                data_clean2.append(item)

        return data_clean2

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_classes2(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes2 = []
        for item in self.data2:
            if item[1] not in classes2:
                classes2.append(item[1])

        # Sort them.
        classes2 = sorted(classes2)

        # Return.
        if self.class_limit is not None:
            return classes2[:self.class_limit]
        else:
            return classes2

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = np_utils.to_categorical(label_encoded, len(self.classes))
        label_hot = label_hot[0]  # just get a single row

        return label_hot

    def get_class_one_hot2(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes2.index(class_str)

        # Now one-hot it.
        label_hot2 = np_utils.to_categorical(label_encoded, len(self.classes))
        label_hot2 = label_hot2[0]  # just get a single row

        return label_hot2

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def split_train_test2(self):
        """Split the data into train and test groups."""
        train2 = []
        test2 = []
        for item in self.data2:
            if item[0] == 'train':
                train2.append(item)
            else:
                test2.append(item)

        return train2, test2

    def get_all_sequences_in_memory(self, batch_size, train_test, data_type, concat=False):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Getting %s data with %d samples." % (train_test, len(data)))

        X, y = [], []
        for row in data:

            sequence = self.get_extracted_sequence(data_type, row)

            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise

            if concat:
                # We want to pass the sequence back as a single array. This
                # is used to pass into a CNN or MLP, rather than an RNN.
                sequence = np.concatenate(sequence).ravel()

            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    def frame_generator(self, batch_size, train_test, data_type, concat=False):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Creating %s generator with %d samples." % (train_test, len(data)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.

                    frames = self.get_frames_for_sample(sample)

                    # frames = self.normalize(frames)

                    if len(frames) <= self.seq_length:
                        continue

                    # sequence = self.rescale_list(frames, self.seq_length)
                    sequence = self.rescale_listpre(frames, self.seq_length)

                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()  # TODO this should raise

                if concat:
                    # We want to pass the sequence back as a single array. This
                    # is used to pass into an MLP rather than an RNN.
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def frame_generator2(self, batch_size, train_test, data_type, concat=False):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test
        train2, test2 = self.split_train_test2()
        data2 = train2 if train_test == 'train' else test2

        print("Creating %s generator with %d samples." % (train_test, len(data2)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample = random.choice(data)
                sample2 = random.choice(data2)

                # Check to see if we've already saved this sequence.
                if data_type is "images":
                    # Get and resample frames.

                    frames = self.get_frames_for_sample(sample)
                    frames2 = self.get_frames_for_sample2(sample2)

                    # 0-1に収めるための処理
                    # frames2 = self.normalize(frames2)

                    if (len(frames) <= self.seq_length) or (len(frames2) <= self.seq_length):
                        continue

                    # sequence = self.rescale_list(frames, self.seq_length)
                    sequence = self.rescale_listpre(frames, self.seq_length)
                    sequence2 = self.rescale_listpre(frames2, self.seq_length)

                else:
                    # Get the sequence from disk.
                    sequence = self.get_extracted_sequence(data_type, sample2)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    sys.exit()  # TODO this should raise

                if concat:
                    # We want to pass the sequence back as a single array. This
                    # is used to pass into an MLP rather than an RNN.
                    sequence = np.concatenate(sequence).ravel()

                X.append([sequence, sequence2])

                y.append([self.get_class_one_hot(sample[1]), self.get_class_one_hot2(sample2[1])])

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample):
        """Get the saved extracted features."""
        filename = sample[2]
        path = self.sequence_path + filename + '-' + str(self.seq_length) + \
               '-' + data_type + '.txt'
        if os.path.isfile(path):
            # Use a dataframe/read_csv for speed increase over numpy.
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = './data/' + sample[0] + '/' + sample[1] + '/'
        filename = sample[2]
        images = np.loadtxt(path + filename + '.csv', delimiter=',', dtype=np.float32).reshape(-1, 16, 16, 1)
        # print('Originalshape is : {}'.format(images.shape))
        images = np.delete(images, 0, 0)
        # print('Deleteshape is : {}'.format(images.shape))
        return images

    @staticmethod
    def get_frames_for_sample2(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = './data/exist_ornot/' + sample[0] + '/' + sample[1] + '/'
        filename = sample[2]
        images = np.loadtxt(path + filename + '.csv', delimiter=',', dtype=np.float32).reshape(-1, 16, 16, 1)
        # print('Originalshape is : {}'.format(images.shape))
        images = np.delete(images, 0, 0)
        # print('Deleteshape is : {}'.format(images.shape))
        return images

    @staticmethod
    def opticalflow(frames):
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=3, blockSize=3)
        lk_params = dict(winSize=(3, 3),
                         maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = np.random.randint(0, 255, (100, 3))

        for i, frame in enumerate(frames):
            tempmax = np.max(frame)
            maxindex = np.argmax(frame)
            y, x = divmod(maxindex, 16)
            tempmin = np.min(frame)
            print(maxindex, x, y)

            pixval = (frame - tempmin) * 255 / (tempmax - tempmin)

            pixval = np.asarray(pixval, dtype=np.uint8)

            cv2.imwrite("pixval{0:03}.jpg".format(i), pixval)
            if i == 0:
                old_gray = pixval
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
                print(p0)
                mask = np.zeros_like(old_gray)
                continue

            frame_gray = pixval
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            if len(p1) > 1:
                good_new = p1[st == 1]
            else:
                good_new = p1[0]
            if len(p0) > 1:
                good_old = p0[st == 1]
            else:
                good_old = p0[0]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1)
                pixval = cv2.circle(pixval, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(pixval, mask)
            cv2.imwrite('opticalFlow{0:03}.jpg'.format(i), mask)
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        sys.exit()

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def rescale_list(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the original list."""
        # print("input.shape=", input_list.shape)

        # assert len(input_list) >= size

        # Get the number to skip between iterations.
        output = input_list[1:]

        if len(output) < size:
            pad = np.zeros(shape=(size - len(output), 16, 16, 1))
            output = np.append(output, np.array(pad), axis=0)

        skip = len(input_list) // size
        # Build our new output.
        # output = output[::skip]
        # Cut off the last one if needed.
        return output[:size]

    @staticmethod
    def rescale_listpre(input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the original list."""
        # assert len(input_list) >= size

        # Get the number to skip between iterations.
        # skip = len(input_list) // size

        # Build our new output.
        output = input_list[1:]
        # Cut off the last one if needed.

        return output[:size]

    @staticmethod
    def print_class_from_prediction(predictions, nb_to_return=5):
        """Given a prediction, print the top classes."""
        # Get the prediction for each label.
        label_predictions = {}
        for i, label in enumerate(data.classes):
            label_predictions[label] = predictions[i]

        # Now sort them.
        sorted_lps = sorted(
            label_predictions.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        # And return the top N.
        for i, class_prediction in enumerate(sorted_lps):
            if i > nb_to_return - 1 or class_prediction[1] == 0.0:
                break
            print("%s: %.2f" % (class_prediction[0], class_prediction[1]))

    @staticmethod
    def normalize(frames):
        pixvals = np.zeros(shape=frames.shape)
        for i, frame in enumerate(frames):
            tempmax = np.max(frame)
            tempmin = np.min(frame)
            pixvals[i] = (frame - tempmin) / (tempmax - tempmin)

        return pixvals
