#%%
from network.gan import GAN
from network.dcgan import DCGAN
from network.conditional_dcgan import CDCGAN
import keras as K 
from keras.optimizers import Adam
import numpy as np
#%%
#gan training
save_path = 'D:\\python\\models\\gan\\test\\img_{}.png'
gan = GAN(784, 100, Adam(0.0002, beta_1 = 0.5))
log = gan.train(epochs = 50, batch_size = 64, display_step = 1, plot_shape = (3, 3), save_path = save_path)
#%%
#print(log)

#dcgan training
#gan = DCGAN((32,32,3), 100, Adam(0.0003, beta_1 = 0.5))
#gan.discriminator.summary()
#gan.generator.summary()
#log = gan.train(epochs = 200, batch_size = 32, display_step = 1, plot_shape = (2, 5), save_path = save_path)

## conditional dcgan training
#gan = CDCGAN((28,28,1), 100, 10, Adam(0.0002, beta_1 = 0.5))
#log = gan.train(epochs = 500, batch_size = 300, display_step = 5, plot_shape = (3, 3))


