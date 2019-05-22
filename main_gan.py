#%%
from network.gan import GAN
from network.dcgan import DCGAN
from network.conditional_dcgan import CDCGAN
import keras as K 
from keras.optimizers import Adam
import numpy as np
#%%
#gan training
#gan = GAN(784, 100, Adam(0.0002, beta_1 = 0.5))
#log = gan.train(epochs = 50, batch_size = 300, display_step = 10, plot_shape = (3, 3))
#%%
#print(log)

#dcgan training
#gan = DCGAN((28,28,1), 100, Adam(0.0002, beta_1 = 0.5))
#log = gan.train(epochs = 500, batch_size = 300, display_step = 5, plot_shape = (3, 3))

## conditional dcgan training
gan = CDCGAN((28,28,1), 100, 10, Adam(0.0002, beta_1 = 0.5))
log = gan.train(epochs = 500, batch_size = 300, display_step = 5, plot_shape = (3, 3))