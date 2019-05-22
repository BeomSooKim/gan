#%%
from keras.layers import Dense, Input, Dropout, Reshape, ReLU, Flatten, GlobalAveragePooling2D
from keras.layers import LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Activation, ELU
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist

import matplotlib.pyplot as plt
#%%
class DCGAN:
    def __init__(self, input_shape, z_shape, optimizer):
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.optimizer = optimizer

        self.discriminator = self.get_discriminator()

        z = Input(shape = (self.z_shape,))
        self.generator = self.get_generator(z)
        
        img = self.generator(z)
        self.discriminator.trainable = False
        fake_img = self.discriminator(img)

        self.adversarial = Model(z, fake_img)
        self.adversarial.compile(loss = 'binary_crossentropy',optimizer = self.optimizer)


    def get_discriminator(self):
        _input = Input(shape = self.input_shape)
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same')(_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        
        x = Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = LeakyReLU(0.3)(x)
        
        x = Conv2D(filters = 256, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        #x = Conv2D(filters = 1, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = Flatten()(x)
        x = Dense(1, activation = 'sigmoid')(x)

        model = Model(_input, x)
        model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return model

    def get_generator(self, _input):
        
        _input = Input(shape = (self.z_shape,))
        x = Dense(7*7*256)(_input)
        x = ReLU()(x)
        x = Reshape((7, 7, 256))(x)
        x = Conv2DTranspose(filters = 128, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters = 1, kernel_size = (3,3),padding = 'same', activation = 'tanh')(x)

        model = Model(_input, x)
        return model

    def show_image(self, savepath, epoch, shape = (2, 5)):
        np.random.seed(13245)
        noise = np.random.normal(0, 0.5, (np.prod(shape), self.z_shape))
        imgs = self.generator.predict(noise)
        imgs = imgs * 127.5 +127.5
        #imgs = imgs.reshape((np.prod(shape), 28, 28))

        fig, axes = plt.subplots(nrows = shape[0],ncols = shape[1], figsize = np.array(shape)[::-1]*2)
        axes = axes.flatten()
        fig.suptitle("{} epochs".format(epoch))
        for (img, ax) in zip(imgs, axes):
            ax.imshow(np.squeeze(img), cmap = 'gray')

        #plt.show()
        fig.savefig(savepath, facecolor = 'w')
        plt.close()
    def train(self, epochs, batch_size, display_step, plot_shape):
        (x_train, _), (_, _) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = x_train.reshape(*x_train.shape, 1)
        #x_train = np.reshape(x_train, (x_train.shape[0], self.input_shape))
        #x_train = np.expand_dims(x_train, -1)
        n_data = len(x_train)
        loss_dict = {"d_loss":[],"d_acc":[], "g_loss":[]}
        for i in np.arange(epochs):
            for _ in np.arange(0, n_data, batch_size):
                idx = np.random.randint(0, high = n_data, size = batch_size)
                # train discriminator
                real_x = x_train[idx]
                real_y = np.ones((batch_size, 1)) - 0.05 *np.random.uniform(0, 1, (batch_size, 1))
                
                z = np.random.uniform(-1, 1, (batch_size, self.z_shape))
                fake_x = self.generator.predict(z)
                fake_y = np.zeros((batch_size, 1)) + 0.05 *np.random.uniform(0, 1, (batch_size, 1))

                # train seperately
                d_loss1 = self.discriminator.train_on_batch(real_x, real_y)
                d_loss2 = self.discriminator.train_on_batch(fake_x, fake_y)
                d_loss = (np.array(d_loss1) +np.array(d_loss2))/2
                # train once
                #all_x = np.concatenate((real_x, fake_x))
                #all_y = np.concatenate((real_y, fake_y))
                #d_loss = self.discriminator.train_on_batch(all_x, all_y)

                #train generator
                #z = np.random.normal(-1, 1, (batch_size, self.z_shape))
                g_loss = self.adversarial.train_on_batch(z, real_y)

                loss_dict['d_loss'].append(d_loss[0])
                loss_dict['d_acc'].append(d_loss[1])
                loss_dict['g_loss'].append(g_loss)
                
            print('{} epoch status :gen_loss = {:.4f} / disc_loss = {:.4f} / disc_acc = {:.4f}'\
               .format(i+1, g_loss, d_loss[0], d_loss[1]))
            if (i + 1) % display_step == 0:
                self.show_image("D:\\study\\4.experiments\\GAN\\images\\dcgan\\{}epoch.png".format(i+1),\
                    i+1, plot_shape)
        return loss_dict

#%%
