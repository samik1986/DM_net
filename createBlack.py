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


fileList1 = os.listdir('membrane/morseUpdate/stp_data/train/img/')
# fileList2 = os.listdir('membrane/morseUpdate/stp_data/train/seg/')
# fileList3 = os.listdir('membrane/morseUpdate/stp_training_morse/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)





def readImagesTwice(files1):
    L = len(files1)
    X = []
    # Y = []
    # Z = []
    # filePath = 'membrane//morseUpdate/'
    for f1 in files1:
        img = np.zeros([512,512])
        # print max(max(row) for row in img)
        img = img.astype('float32')
        # print filePath + name2 + "/" + f1[:-8] + '_mask.tif', filePath + name1 + "/" + f1
        # mask = misc.imread(filePath + name2 + "/" + f1.replace("row", "label")).astype('float32')
        # dm = misc.imread(filePath + name3 + "/" + f1).astype('float32')
        # if img.max():
        #     img = img / img.max()
        # dm = dm / 255.
        # mask = mask * 255.
        # cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # cv2.normalize(dm.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # cv2.normalize(mask.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # print max(max(row) for row in img)
        # print min(min(row) for row in img)
        # case = np.random.randint(1,4)
        # if case == 1:
        #     val = np.random.randint(-1,1)
        #     img = cv2.flip(img, val)
        #     dm = cv2.flip(dm, val)
        #     mask = cv2.flip(mask, val)
        # if case == 2:
        #     val = np.random.uniform(0.001, 0.2)
        #     img = transform.warp(img, inverse_map=transform.AffineTransform(shear=val))
        #     dm = transform.warp(dm, inverse_map=transform.AffineTransform(shear=val))
        #     mask = transform.warp(mask, inverse_map=transform.AffineTransform(shear=val))
        # if case == 3:
        #     val = np.random.randint(-15,15)
        #     img = ndimage.rotate(img, val, reshape=False)
        #     dm = ndimage.rotate(dm, val, reshape=False)
        #     mask = ndimage.rotate(mask, val, reshape=False)
        # if case == 4:
        #     val = np.random.uniform(0.5, 2)
        #     img = clipped_zoom(img, val)
        #     dm = clipped_zoom(dm, val)
        #     mask = clipped_zoom(mask, val)
        # img = (img.astype(np.float32)-127.5)/127.5
        X.append(img)
        # Y.append(dm)
        # Z.append(mask)
    X_arr = np.asarray(X)
    X_arr = X_arr[..., np.newaxis]
    # Y_arr = np.asarray(Y)
    # Y_arr = Y_arr[..., np.newaxis]
    # Z_arr = np.asarray(Z)
    # Z_arr = Z_arr[..., np.newaxis]

    return X_arr#, Y_arr, Z_arr


# def imageLoader(files1, batch_size):
#
#     L = len(files1)
#
#     #this line is just to make the generator infinite, keras needs that
#     while True:
#
#         batch_start = 0
#         batch_end = batch_size
#
#         while batch_start < L:
#             limit = min(batch_end, L)
#             # print type(files1[batch_start:limit])
#             [X, Y, Z] = readImagesTwice(files1,'stp_data/train/img', 'stp_data/train/seg', 'stp_training_morse')
#             # Y = readImages(files2[batch_start:limit], '405_imgs')
#
#             yield (X,Y, Z) #a tuple with two numpy arrays with batch_size samples
#
#             batch_start += batch_size
#             batch_end += batch_size
#

X = readImagesTwice(fileList1)
# np.save('dm.npy', Y)
np.save('imgBlk.npy', X)
# np.save('segTrn.npy', Z)

