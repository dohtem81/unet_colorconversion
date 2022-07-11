from skimage import color
from skimage.io import imread, imsave
from skimage.transform import resize
import os
from tqdm import tqdm

# go to image holding folder
os.chdir("../../imgs")

# go over each file read it, resize it, save as color one, save as BW one
for n, imageFile in tqdm(enumerate(os.listdir("./originals")), total=len(os.listdir("./originals"))):  
    print('working on ' + imageFile)
    imageOriginal = imread('./originals/' + imageFile)
    imgageColorResized = resize(imageOriginal, (512, 512))
    imageBWResized = color.rgb2gray(imageOriginal)
    imageBWResized = resize(imageBWResized, (512, 512))
    imsave('./color/'+imageFile, imgageColorResized)
    imsave('./bandw/'+imageFile, imageBWResized)
