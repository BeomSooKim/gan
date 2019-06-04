#%%
import imageio
from imutils.paths import list_images
import numpy as np 
images =[]

img_paths = list_images("D:\\python\\models\\gan\\test_noBN")
img_paths = np.array(list(img_paths))
print(img_paths)

for f in img_paths:
    images.append(imageio.imread(f))
imageio.mimsave('.\\images\\gan_noBN.gif', images, duration = 0.5)