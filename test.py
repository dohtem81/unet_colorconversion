import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import models

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1

# location of the images
SOURCE_PATH = '../../imgs/bandw'
COMP_PATH = '../../imgs/color'
DEST_PATH = '../../imgs/test'

X_test = np.zeros((len(os.listdir(SOURCE_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

train_ids = next(os.walk(SOURCE_PATH))[2]

print('reading test set')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = SOURCE_PATH + '/' + id_
    inimg = imread(path)
    inimg = np.expand_dims(resize(inimg, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
    X_test[n] = inimg
print('done')

# read model
mdl = models.unet(pretrained_model='model_gray2rgb.h5', input_channel = 1, output_channnels = 3)
results = mdl.predict(X_test)

for n, imgArray in tqdm(enumerate(results), total=len(results)):   
    path = DEST_PATH + '/gray2rgb_' + str(n) + '.jpg'
    imsave(path, imgArray)