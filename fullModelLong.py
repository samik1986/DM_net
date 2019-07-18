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
from createNEt_long import *

fileList1 = os.listdir('membrane/morseUpdate/stp_data/train/img/')
fileList2 = os.listdir('membrane/morseUpdate/stp_data/train/seg/')
fileList3 = os.listdir('membrane/morseUpdate/stp_training_morse/')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)



# def readImagesTwiceV(files1, name1, name2, name3):
#     L = len(files1)
#     X = []
#     Y = []
#     filePath = 'membrane/morseUpdate/'
#     for f1 in files1:
#         img = misc.imread(filePath + name1 + "/" + f1).astype('float32')
#         # print filePath + name2 + "/" + f1[:-8] + '_mask.tif', filePath + name1 + "/" + f1
#         mask = misc.imread(filePath + name2 + "/" + f1).astype('float32')
#         dm = misc.imread(filePath + name2 + "/" + f1).astype('float32')
#         cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         cv2.normalize(dm.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         cv2.normalize(mask.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         case = np.random.randint(1, 4)
#         if case == 1:
#             val = np.random.randint(-1, 1)
#             img = cv2.flip(img, val)
#             dm = cv2.flip(dm, val)
#             mask = cv2.flip(mask, val)
#         if case == 2:
#             val = np.random.uniform(0.001, 0.2)
#             img = transform.warp(img, inverse_map=transform.AffineTransform(shear=val))
#             dm = transform.warp(dm, inverse_map=transform.AffineTransform(shear=val))
#             mask = transform.warp(mask, inverse_map=transform.AffineTransform(shear=val))
#         if case == 3:
#             val = np.random.randint(-15, 15)
#             img = ndimage.rotate(img, val, reshape=False)
#             dm = ndimage.rotate(dm, val, reshape=False)
#             mask = ndimage.rotate(mask, val, reshape=False)
#         if case == 4:
#             val = np.random.uniform(0.5, 2)
#             img = clipped_zoom(img, val)
#             dm = clipped_zoom(dm, val)
#             mask = clipped_zoom(mask, val)
#         # img = (img.astype(np.float32)-127.5)/127.5
#         X.append(img)
#         Y.append(dm)
#         Z.append(mask)
#     X_arr = np.asarray(X)
#     X_arr = X_arr[..., np.newaxis]
#     Y_arr = np.asarray(Y)
#     Y_arr = Y_arr[..., np.newaxis]
#     Z_arr = np.asarray(Z)
#     Z_arr = Z_arr[..., np.newaxis]
#     return X_arr, Y_arr, Z_arr
#
#
# def readImagesTwice(files1, name1, name2, name3):
#     L = len(files1)
#     X = []
#     Y = []
#     Z = []
#     filePath = 'membrane/morseUpdate/'
#     for f1 in files1:
#         img = misc.imread(filePath + name1 + "/" + f1).astype('float32')
#         # print filePath + name2 + "/" + f1[:-8] + '_mask.tif', filePath + name1 + "/" + f1
#         mask = misc.imread(filePath + name2 + "/" + f1.replace("row","label")).astype('float32')
#         dm = misc.imread(filePath + name3 + "/" + f1).astype('float32')
#         cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         cv2.normalize(dm.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         cv2.normalize(mask.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
#         cv2.resize(img, 0.5)
#         case = np.random.randint(1,4)
#         if case == 1:
#             val = np.random.randint(-1,1)
#             img = cv2.flip(img, val)
#             dm = cv2.flip(dm, val)
#             mask = cv2.flip(mask, val)
#         if case == 2:
#             val = np.random.uniform(0.001, 0.2)
#             img = transform.warp(img, inverse_map=transform.AffineTransform(shear=val))
#             dm = transform.warp(dm, inverse_map=transform.AffineTransform(shear=val))
#             mask = transform.warp(mask, inverse_map=transform.AffineTransform(shear=val))
#         if case == 3:
#             val = np.random.randint(-15,15)
#             img = ndimage.rotate(img, val, reshape=False)
#             dm = ndimage.rotate(dm, val, reshape=False)
#             mask = ndimage.rotate(mask, val, reshape=False)
#         if case == 4:
#             val = np.random.uniform(0.5, 2)
#             img = clipped_zoom(img, val)
#             dm = clipped_zoom(dm, val)
#             mask = clipped_zoom(mask, val)
#         # img = (img.astype(np.float32)-127.5)/127.5
#         X.append(img)
#         Y.append(dm)
#         Z.append(mask)
#     X_arr = np.asarray(X)
#     X_arr = X_arr[..., np.newaxis]
#     Y_arr = np.asarray(Y)
#     Y_arr = Y_arr[..., np.newaxis]
#     Z_arr = np.asarray(Z)
#     Z_arr = Z_arr[..., np.newaxis]
#     return X_arr, Y_arr, Z_arr
#
#
# def clipped_zoom(img, zoom_factor, **kwargs):
#
#     h, w = img.shape[:2]
#
#     # For multichannel images we don't want to apply the zoom factor to the RGB
#     # dimension, so instead we create a tuple of zoom factors, one per array
#     # dimension, with 1's for any trailing dimensions after the width and height.
#     zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
#
#     # Zooming out
#     if zoom_factor < 1:
#
#         # Bounding box of the zoomed-out image within the output array
#         zh = int(np.round(h * zoom_factor))
#         zw = int(np.round(w * zoom_factor))
#         top = (h - zh) // 2
#         left = (w - zw) // 2
#
#         # Zero-padding
#         out = np.zeros_like(img)
#         out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)
#
#     # Zooming in
#     elif zoom_factor > 1:
#
#         # Bounding box of the zoomed-in region within the input array
#         zh = int(np.round(h / zoom_factor))
#         zw = int(np.round(w / zoom_factor))
#         top = (h - zh) // 2
#         left = (w - zw) // 2
#
#         out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
#
#         # `out` might still be slightly larger than `img` due to rounding, so
#         # trim off any extra pixels at the edges
#         trim_top = ((out.shape[0] - h) // 2)
#         trim_left = ((out.shape[1] - w) // 2)
#         out = out[trim_top:trim_top+h, trim_left:trim_left+w]
#
#     # If zoom_factor == 1, just return the input array
#     else:
#         out = img
#     return out
#
#
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
#             [X, Y, Z] = readImagesTwice(files1[batch_start:limit],'stp_data/train/img', 'stp_data/train/seg', 'stp_training_morse')
#             # Y = readImages(files2[batch_start:limit], '405_imgs')
#
#             yield ([X, Y], Z) #a tuple with two numpy arrays with batch_size samples
#
#             batch_start += batch_size
#             batch_end += batch_size
#
# def imageLoaderV(files1, batch_size):
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
#             [X, Y, Z] = readImagesTwice(files1, 'stp_data/train/img', 'stp_data/train/seg', 'stp_training_morse')
#             # Y = readImages(files2[batch_start:limit], '405_imgs')
#
#             yield (X, Y, Z) #a tuple with two numpy arrays with batch_size samples
#
#             batch_start += batch_size
#             batch_end += batch_size
#
# # data_gen_args = dict(rotation_range=0.2,
# #                     horizontal_flip=True,
# #                     fill_mode='nearest')
# # myGene = trainGenerator(5,'membrane/train','image','label',data_gen_args,save_to_dir = None)
# # validation = validateGenerator('membrane/validate/', num_image=1155)

X = np.load("imgTrn.npy")
# Y = np.load("dm.npy")
Z = np.load("segTrn.npy")

model = dmnetLong()
print model.summary()
# model.load_weights('unet_membrane.hdf5')
model_checkpoint = ModelCheckpoint('dmnet_membraneLong.hdf5', monitor='loss',verbose=1, save_best_only=True)
tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X, Z, epochs=50, batch_size= 20, callbacks=[model_checkpoint,tbCallBack])# ,validation_data=imageLoaderV(fileList1, 2),validation_steps=578)

