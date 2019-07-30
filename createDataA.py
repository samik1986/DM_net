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


fileList1 = os.listdir('membrane/morseUpdate/train_pred_albu/')
fileList2 = os.listdir('membrane/morseUpdate/stp_data/train/seg/')
fileList3 = os.listdir('membrane/morseUpdate/stp_training_morse/')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



def readImagesTwice(files1, name1, name2, name3):
    L = len(files1)
    X = []
    Y = []
    Z = []
    filePath = 'membrane/morseUpdate/'
    for f1 in files1:
        img = misc.imread(filePath + name1 + "/" + f1)
        # print max(max(row) for row in img)
        img = img.astype('float32')
        # print filePath + name2 + "/" + f1[:-8] + '_mask.tif', filePath + name1 + "/" + f1
        mask = misc.imread(filePath + name2 + "/" + f1.replace("img", "label")).astype('float32')
        dm = misc.imread(filePath + name3 + "/" + f1.replace("img", "row")).astype('float32')
        if img.max():
            img = img / img.max()
        dm = dm / 255.
        X.append(img)
        Y.append(dm)
        Z.append(mask)
    X_arr = np.asarray(X)
    X_arr = X_arr[..., np.newaxis]
    Y_arr = np.asarray(Y)
    Y_arr = Y_arr[..., np.newaxis]
    Z_arr = np.asarray(Z)
    Z_arr = Z_arr[..., np.newaxis]

    return X_arr, Y_arr, Z_arr


[X, Y, Z] = readImagesTwice(fileList1,'train_pred_albu', 'stp_data/train/seg', 'stp_training_morse')
np.save('dm1.npy', Y)
np.save('imgAlbuTrn.npy', X)
np.save('segTrn.npy', Z)

