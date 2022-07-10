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
IMG_CHANNELS = 3

# location of the images
TRAIN_PATH = '../../imgs/color'
MASK_PATH = '../../imgs/bandw'
TEST_PATH = '../../imgs/test'

train_ids = next(os.walk(TRAIN_PATH))[2]

# prerp source arrays
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=float)

print('reading training set')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + '/' + id_
    img = imread(path)[:,:,:IMG_CHANNELS]  
    X_train[n] = img
    outmsg = imread(MASK_PATH + '/' + id_) / 255
    Y_train[n] = np.expand_dims(resize(outmsg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)

print('done')

# get modelmd
mdl = models.unet()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_rgb2grayscale_best.h5', verbose=1, save_best_only=True)
callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')]
####################################

results = mdl.fit(X_train, Y_train, validation_split=0.1, batch_size=4, epochs=25, callbacks=callbacks)
mdl.save('model_rgb2grayscale.h5')
