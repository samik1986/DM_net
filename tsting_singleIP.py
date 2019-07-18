import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, Model, save_model
# import Image
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
from modelUnet import *
from data import *
import cv2
import tensorflow as tf
from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
from keras.callbacks import TensorBoard
from scipy import misc
from createNEt import *

# model = dmnet()

model = load_model('dmnet_membraneLong.hdf5')

# X = np.load("imgTst.npy")
# Y = np.load("dmTST.npy")

fileList1 = os.listdir('membrane/morseUpdate/stp_data/test/img/')
fileList2 = os.listdir('membrane/morseUpdate/stp_data/test/seg/')
fileList3 = os.listdir('membrane/morseUpdate/stp_test_morse/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)


def testImages(files1, name1, name2):
    L = len(files1)
    X = []
    Y = []
    filePath = 'membrane/morseUpdate/'
    for f1 in files1:
        img = misc.imread(filePath + name1 + "/" + f1)
        # print max(max(row) for row in img)
        img = img.astype('float32')
        # print filePath + name2 + "/" + f1[:-8] + '_mask.tif', filePath + name1 + "/" + f1
        # mask = misc.imread(filePath + name2 + "/" + f1.replace("row", "label")).astype('float32')
        # dm = misc.imread(filePath + name2 + "/" + f1).astype('float32')
        if img.max():
            img = img / img.max()
        # dm = dm / 255.

        X_arr = np.asarray(img)
        X_arr = X_arr[..., np.newaxis]
        X_arr = X_arr[np.newaxis, ...]
        print X_arr.shape
        # Y_arr = np.asarray(dm)
        # Y_arr = Y_arr[..., np.newaxis]
        # Y_arr = Y_arr[np.newaxis, ...]
        out_img = model.predict(X_arr)

        img = np.squeeze(out_img[0]) * 255.
        misc.imsave(filePath + "resultsLong/" + f1, img.astype('uint8'))
        # mask = mask * 255.
        # cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # cv2.normalize(dm.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # cv2.normalize(mask.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # print max(max(row) for row in img)
        # print min(min(row) for row in img)
        # img = (img.astype(np.float32)-127.5)/127.5
    #     X.append(img)
    #     Y.append(dm)
    #     Z.append(mask)
    # X_arr = np.asarray(X)
    # X_arr = X_arr[..., np.newaxis]
    # Y_arr = np.asarray(Y)
    # Y_arr = Y_arr[..., np.newaxis]
    # Z_arr = np.asarray(Z)
    # Z_arr = Z_arr[..., np.newaxis]

    # return X_arr, Y_arr


testImages(fileList1,'stp_data/test/img', 'stp_test_morse')