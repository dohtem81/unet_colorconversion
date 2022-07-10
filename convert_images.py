from skimage import color
from skimage.transform import resize
import os

# go to image holding folder
os.chdir("../../imgs")

# go over each file read it, resize it, save as color one, save as BW one
for imageFile in os.listdir("./originals"):
    imageOriginal = skimage.io.imread('./originals/' + imageFile)
    imgageColorResized = resize(imageOriginal, (512, 512))
    imageBWResized = color.rgb2gray(imageOriginal)
    imageBWResized = resize(imageBWResized, (512, 512))
    skimage.io.imsave('./color/'+imageFile, imgageColorResized)
    skimage.io.imsave('./bandw/'+imageFile, imageBWResized)
