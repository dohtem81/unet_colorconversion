<h1>Set of interesting experiments</h1>
There are 2 models- gray to color and color to gray. To be honest these models should not be commited here, but they ended up here.

<h2>What it does</h2>
Models are converting color images into grayscale (yeah.. I know, there are easier ways to do that), although it was first step when I was fighting COVID - do something useful when sentenced to stay in one room for few days. Converting from color to rayscale seamed to be much simplier task and given I did not have GPU handly, easier to train the model. 

Why easier?

For many color combinations we end up with same number in grayscale image. To train model it would take much less time to get the results (or not) and prove it can work. Idea was to use segmentation model and basically instead of training it for binary 0-1 answer, train it to give values of the pixels. In theory it sounded it could work... and it did.

Once proven easy task, why not to try to color grayscale images (idea came because I was actually watching WW2 documentary on Netflix where videos were colored). And again - this has been done many times, many papaers about it, but the challange for me was "can I figure it out without reading any papers about how it can be done". 

Big changes to the model - instead of 3 channels in, 1 channel out, there is 1 channel in, 3 channels out. This time from simple RGB values = GRAY value, more magic had to happen. Model had to extract features like sand, water, faces, clothes and assign colors. That did sound wild, and initially it did not work. Why? Learning too slow using CPU only, but not having NVidia GPU did not offer me any solution. Until...

I'm using Macbook Pro with AMD GPU and I was able to find article how to use force Keras to use CUDA on AMD (more then super simplification), but bottom line, with couple tricks and packages installed, AMD GPU started to work with my code. 

These are results:] of feeding grayscale images to the model and getting out of it RGB ones (first link is coolest one):
https://www.linkedin.com/posts/piotr-pedziwiatr_machinelearning-video-creativity-activity-6978247420883202048-U8Rt?utm_source=share&utm_medium=member_desktop

https://www.linkedin.com/posts/piotr-pedziwiatr_ai-software-covid-activity-6972440328993730561-0uvh?utm_source=share&utm_medium=member_desktop

https://www.linkedin.com/posts/piotr-pedziwiatr_ai-software-covid-activity-6972019922524663808-Wzh0?utm_source=share&utm_medium=member_desktop
