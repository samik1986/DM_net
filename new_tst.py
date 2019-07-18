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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc


X = np.load("imgTrn.npy")
Y = np.load("dm.npy")
Z = np.load("segTrn.npy")

misc.imsave("img.jp2", np.squeeze(X[3]))
misc.imsave("dm.jp2", np.squeeze(Y[3]))
misc.imsave("seg.jp2", np.squeeze(Z[3]))

# # print(np.squeeze(X[3]).sha
# fig = plt.figure()
# ax1 = fig.add_subplot(1,3,1)
# ax1.imshow(np.squeeze(X[3]))
# ax2 = fig.add_subplot(1,3,2)
# ax2.imshow(np.squeeze(Y[3]))
# ax3 = fig.add_subplot(1,3,3)
# ax3.imshow(np.squeeze(Z[3]))
# # axarr[1,1].imshow(image_datas[3])
#
#
# # plt.show()
# plt.savefig('hlp.png')

