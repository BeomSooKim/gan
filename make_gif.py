#%%
import imageio
from imutils.paths import list_images
import numpy as np 
images =[]

img_paths = list_images("D:\\study\\4.experiments\\GAN\\images")
img_paths = np.array(list(img_paths))
print(img_paths)

for f in img_paths:
    images.append(imageio.imread(f))
imageio.mimsave('.\\images\\generator.gif', images)