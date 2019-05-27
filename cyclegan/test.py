#%%
from keras.models import load_model
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
import os
from IPython.display import display
#%%
model_GA = load_model('./weights-cyclelossweight10-batchsize16-imagesize128/tf_G_A_model200.hdf5')
model_GB = load_model('./weights-cyclelossweight10-batchsize16-imagesize128/tf_G_B_model200.hdf5')
#%%
test_image = Image.open("D:\\python\\dataset\\horse2zebra\\testA\\n02381460_840.jpg")
test_image = test_image.resize((128, 128))
display(test_image)
test_arr = np.array(test_image)
test_arr = (test_arr - 127.5) / 127.5
test_arr = np.expand_dims(test_arr, 0)
test_arr.shape

pred = model_GA.predict(test_arr)
#%%
def reconstruct(img):
    img = img * 127.5 + 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

pred = np.squeeze(pred)
pred_recon = reconstruct(pred)

recon_img = Image.fromarray(pred_recon)
display(recon_img)