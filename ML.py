import pandas as pd
import numpy as np
import pickle
import glob
import json
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

# import joblib

dif = "NAe"
Ans = "Nobody"
Pre = "AreaEntry"


class Logknn(KNeighborsClassifier):
    # Override the class constructor
    def __init__(self, data=None, nb_class=2, labels=None, k_neighbors=8):
        KNeighborsClassifier.__init__(self, n_neighbors=k_neighbors)
        self.data = data
        self.labels = labels
        self.nb_class = nb_class
        self.k_neighbors = k_neighbors

    # A method for saving object data to JSON file
    def save_json(self, filepath):
        dict_ = {}
        dict_['nb_class'] = self.nb_class
        dict_['k_neighbors'] = self.k_neighbors
        dict_['data'] = self.data.tolist() if self.data is not None else 'None'
        dict_['labels'] = self.labels.tolist() if self.labels is not None else 'None'

        # Creat json and save to file
        json_txt = json.dumps(dict_, indent=4)
        with open(filepath, 'w') as file:
            file.write(json_txt)

    # A method for loading data from JSON file
    def load_json(self, filepath):
        with open(filepath, 'r') as file:
            dict_ = json.load(file)

        self.nb_class = dict_['nb_class']
        self.k_neighbors = dict_['k_neighbors']
        self.data = np.asarray(dict_['data']) if dict_['data'] != 'None' else None
        self.labels = np.asarray(dict_['labels']) if dict_['labels'] != 'None' else None


class Clustering():
    def __init__(self, data=None, nb_class=2, labels=None, k_neighbors=8):
        self.data = data
        self.nb_class = nb_class
        self.labels = labels
        self.k_neighbors = k_neighbors

    def kmeans(self):
        model = KMeans(n_clusters=self.nb_class)
        predictor = model.fit(self.data)
        filename = "KMeans-" + dif + "-.sav"
        joblib.dump(model, filename, compress=True)
        return predictor

    def svm(self):
        model = svm.SVC()
        predictor = model.fit(self.data, self.labels)
        filename = "SVC-" + dif + "-.sav"
        joblib.dump(model, filename, compress=True)
        return predictor

    def knn(self):
        model = KNeighborsClassifier(n_neighbors=self.k_neighbors)
        predictor = model.fit(self.data, self.labels)
        filename = "KNN5-" + dif + "-.sav"
        # file = open(filename, 'wb')
        # pickle.dump(model, file, protocol=2)
        # file.close()

        joblib.dump(model, filename, compress=True)
        return predictor


def train():
    trainpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/train/"
    trainclasses = sorted(glob.glob(trainpath + '*'))
    nb_trainfile = 0
    trainfiles = []
    for trainclass in trainclasses:
        trainclassname = trainclass.split('/')[-1]
        filepath = glob.glob(trainclass + '/temp*')
        trainfiles.extend(filepath)

    data, labels = [], []
    for trainfile in trainfiles:
        trainclass = trainfile.split('/')[-2]
        temps = np.loadtxt(trainfile, delimiter=',')
        px, py = [], []
        for i, frame in enumerate(temps):
            maxtemp = np.argmax(frame)
            sumtemp = np.sum(frame)
            y, x = divmod(maxtemp, 16)
            px.append(x)
            py.append(y)
            if i < 10:
                continue
            else:
                mx10 = x - px[0]
                my10 = y - py[0]
                mx1 = x - px[-1]
                my1 = y - py[-1]
                del px[0]
                del py[0]

            if trainclass == Ans:
                label = 1
            # elif trainclass == ('AreaExit' or 'AreaEntry'):
            #     continue
            elif trainclass == Pre:
                label = 0
            else:
                continue

            # data.append([frame[maxtemp], sumtemp, mx, my])
            data.append([frame[maxtemp], sumtemp, maxtemp, mx1, my1, mx10, my10])
            labels.append(label)
            nb_trainfile += 1
    data = np.array(data)
    labels = np.array(labels)
    cls = Clustering(data=data, labels=labels, nb_class=2, k_neighbors=5)
    mylog = Logknn(data=data, labels=labels, nb_class=2, k_neighbors=5)
    mylog.save_json("knnlog-NAe-.json")
    print(nb_trainfile, 'trainfiles')
    return cls


def main():
    cls = train()

    pred1 = cls.kmeans()
    pred2 = cls.svm()
    pred3 = cls.knn()

    testpath = "/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/test/"
    testfiles = sorted(glob.glob(testpath + '*'))

    count = 0.0
    kmc, sc, knc, knvmc = 0.0, 0.0, 0.0, 0.0

    for testfile in testfiles:
        files = glob.glob(testfile + "/temp*")

        testclass = testfile.split('/')[-1]

        for file in files:
            temps = np.loadtxt(file, delimiter=',')
            px, py = [], []

            for i, frame in enumerate(temps):
                maxtemp = np.argmax(frame)
                sumtemp = np.sum(frame)
                y, x = divmod(maxtemp, 16)
                px.append(x)
                py.append(y)

                if i < 10:
                    continue
                else:
                    mx10 = x - px[0]
                    my10 = y - py[0]
                    mx1 = x - px[-1]
                    my1 = y - py[-1]
                    del px[0]
                    del py[0]

                if testclass == Ans:
                    label = 1
                elif testclass == Pre:
                    label = 0
                else:
                    continue

                x = np.array([frame[maxtemp], sumtemp, maxtemp, mx1, my1, mx10, my10])
                kms = pred1.predict([x])[0]
                svc = pred2.predict([x])[0]
                knn3 = pred3.predict([x])[0]
                knvm = svc and knn3

                if kms == label:
                    kmc += 1
                if svc == label:
                    sc += 1
                if knn3 == label:
                    knc += 1
                if knvm == label:
                    knvmc += 1
                count += 1
    print("KMeans Accuracy", kmc / count)
    print("SVM Accuracy", sc / count)
    print("KNN_5 Accuracy", knc / count)
    print("KNVM Accuracy", knvmc / count)
    print(count, 'files')

    # jsonlogreg = Logknn()
    # jsonlogreg.load_json("/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/knnlog-NAe-.json")
    # jsonlogreg()


if __name__ == '__main__':
    main()
