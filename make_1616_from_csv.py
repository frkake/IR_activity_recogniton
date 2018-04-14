import cv2
import numpy as np
import os
import csv


def main():
    trainfile_list = []
    testfile_list = []

    TRAIN_PATH = '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/train'
    TEST_PATH = '/home/takubuntu/PycharmProjects/DL/Wake_detect/IR_classification/data/test'

    traindir_list = os.listdir(TRAIN_PATH)
    testdir_list = os.listdir(TEST_PATH)

    for traindir in traindir_list:
        for file in os.listdir(os.path.join(TRAIN_PATH, traindir)):
            if file[:3] == 'pix':
                filenp = np.loadtxt(os.path.join(TRAIN_PATH, traindir, file), delimiter=',', dtype=np.float32).reshape(-1, 16, 16)
                fourcc = cv2.VideoWriter_fourcc(*'FLV1')
                rec = cv2.VideoWriter(TRAIN_PATH + '/' + traindir + '/16x16_{}.avi'.format(file[6:-4]), fourcc, 3.0, (16, 16))
                for i in range(len(filenp)):
                    # cv2.imwrite(TRAIN_PATH + '/' + traindir + "/16x16_" + file[6:-4] + '--{}.jpg'.format(i), filenp[i])
                    # img = cv2.imread(TRAIN_PATH + '/' + traindir + '/16x16_' + file[6:-4] + '--{}.jpg'.format(i))
                    print(filenp[i].shape)
                    rec.write(filenp[i])
                    # cv2.imwrite(TRAIN_PATH+'/'+traindir+"/16x16_"+file[6:-4]+'--{}.jpg'.format(i), filenp[i])
                rec.release()

                # for testdir in testdir_list:
                #     for file in os.listdir(os.path.join(TEST_PATH, traindir)):
                #         if file[:3] == 'pix':
                #             filenp = np.loadtxt(os.path.join(TEST_PATH, traindir, file), delimiter=',').reshape(-1,16,16)
                #             print(file)
                #             print(filenp)
                #             for i in range(len(filenp)):
                #                 cv2.imwrite(TEST_PATH+'/'+traindir+"/16x16_"+file[6:-4]+'--{}.jpg'.format(i), filenp[i])


if __name__ == '__main__':
    main()
