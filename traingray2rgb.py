import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import models

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

# location of the images
TRAIN_PATH = '../../imgs/bandw'
MASK_PATH = '../../imgs/color'
TEST_PATH = '../../imgs/test'

train_ids = next(os.walk(TRAIN_PATH))[2]

# prerp source arrays
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=float)

print('reading training set')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + '/' + id_
    inimg = imread(path)
    inimg = np.expand_dims(resize(inimg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    X_train[n] = inimg
    #outmsg = imread(MASK_PATH + '/' + id_) / 255
    #Y_train[n] = np.expand_dims(resize(outmsg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    Y_train[n] = imread(MASK_PATH + '/' + id_) / 255

print('done')

# get modelmd
mdl = models.unet(pretrained_model='model_gray2rgb.h5', input_channel = 1, output_channnels = 3)

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_gray2rgb_best.h5', verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss'), 
    tf.keras.callbacks.ModelCheckpoint(filepath='model.gray2rgb.{epoch:02d}.h5', save_freq=10)]
####################################

results = mdl.fit(X_train, Y_train, validation_split=0.2, batch_size=5, epochs=100, callbacks=callbacks)
mdl.save('model_gray2rgb.h5')
