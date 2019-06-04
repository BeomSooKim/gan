#%%
from keras.layers import Dense, Input, Dropout, Reshape, ReLU, Flatten, GlobalAveragePooling2D
from keras.layers import LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Activation, ELU
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist, cifar10
from keras import initializers

import matplotlib.pyplot as plt
#%%
class DCGAN:
    def __init__(self, input_shape, z_shape, optimizer):
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.optimizer = optimizer
        self.init = initializers.RandomNormal(stddev = 0.02)

        self.discriminator = self.get_discriminator()
        self.discriminator.summary()
        z = Input(shape = (self.z_shape,))
        self.generator = self.get_generator(z)
        self.generator.summary()
        
        img = self.generator(z)
        self.discriminator.trainable = False
        fake_img = self.discriminator(img)

        self.adversarial = Model(z, fake_img)
        self.adversarial.compile(loss = 'binary_crossentropy',optimizer = self.optimizer)

    def get_discriminator(self):
        _input = Input(shape = self.input_shape)
        x = Conv2D(filters = 64, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = self.init)(_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        
        x = Conv2D(filters = 128, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU(0.2)(x)
        
        x = Conv2D(filters = 256, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU(0.2)(x)
        x = Conv2D(filters = 512, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        #x = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), padding = 'same')(x)
        #x = BatchNormalization()(x)
        #x = LeakyReLU(0.2)(x)
        #x = Conv2D(filters = 1, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = Flatten()(x)
        x = Dense(1, activation = 'sigmoid')(x)

        model = Model(_input, x)
        model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return model

    def get_generator(self, _input):
        
        _input = Input(shape = (self.z_shape,))
        x = Dense(2*2*512, kernel_initializer = self.init)(_input)
        x = ReLU()(x)
        x = Reshape((2, 2, 512))(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(filters = 256, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 128, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 64, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 3, kernel_size = (5,5), strides = (2,2), activation = 'tanh', padding = 'same', use_bias = False)(x)
        #x = Conv2DTranspose(filters = 3, kernel_size = (5,5), strides = (2,2), padding = 'same', use_bias = False)(x)
        #x = BatchNormalization()(x)
        #x = BatchNormalization()(x)
        #x = ReLU()(x)
        #x = Conv2D(filters = 3, kernel_size = (3,3),padding = 'same', activation = 'tanh')(x)

        model = Model(_input, x)
        return model

    def show_image(self, savepath, epoch, shape = (2, 5)):
        #np.random.seed(13245)
        noise = np.random.normal(0, 1, (np.prod(shape), self.z_shape))
        imgs = self.generator.predict(noise)
        imgs = imgs * 127.5 +127.5
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)
        #imgs = imgs.reshape((np.prod(shape), 28, 28))

        fig, axes = plt.subplots(nrows = shape[0],ncols = shape[1], figsize = np.array(shape)[::-1]*0.8)
        axes = axes.flatten()
        fig.suptitle("{} epochs".format(epoch))
        for (img, ax) in zip(imgs, axes):
            ax.imshow(np.squeeze(img), cmap = 'gray')
            ax.axis('off')
        #plt.tight_layout()
        #plt.show()
        fig.savefig(savepath, facecolor = 'w')
        plt.close()
    def train(self, epochs, batch_size, display_step, plot_shape, save_path):
        (x_train, y_train), (_, _) = cifar10.load_data()
        selected = np.where(y_train == 0)[0]
        x_train = x_train[selected,:,:,:]
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        #x_train = x_train.reshape(*x_train.shape, 1)
        #x_train = np.reshape(x_train, (x_train.shape[0], self.input_shape))
        #x_train = np.expand_dims(x_train, -1)
        n_data = len(x_train)
        loss_dict = {"d_loss":[],"d_acc":[], "g_loss":[]}
        for i in np.arange(epochs):
            for _ in np.arange(0, n_data, batch_size):
                idx = np.random.randint(0, high = n_data, size = batch_size)
                # train discriminator
                real_x = x_train[idx]
                z = np.random.normal(0, 1, (batch_size, self.z_shape))
                fake_x = self.generator.predict(z)
                #real_y = np.ones(batch_size) - 0.1
                #fake_y = np.zeros(batch_size)
                all_x = np.concatenate((real_x, fake_x))
                all_y = np.zeros(2*batch_size)
                all_y[:batch_size] = 0.9 #for gradient, set real label to 0.9
                # train seperately
                #self.discriminator.trainable = True
                #d_loss1 = self.discriminator.train_on_batch(real_x, real_y)
                #d_loss2 = self.discriminator.train_on_batch(fake_x, fake_y)
                #d_loss = (np.array(d_loss1) +np.array(d_loss2))/2
                # train once
                d_loss = self.discriminator.train_on_batch(all_x, all_y)

                #train generator
                #z = np.random.normal(-1, 1, (batch_size, self.z_shape))
                #self.discriminator.trainable = False
                g_loss = self.adversarial.train_on_batch(z, np.ones(batch_size))

                loss_dict['d_loss'].append(d_loss[0])
                loss_dict['d_acc'].append(d_loss[1])
                loss_dict['g_loss'].append(g_loss)
                
            print('{} epoch status :gen_loss = {:.4f} / disc_loss = {:.4f} / disc_acc = {:.4f}'\
               .format(i+1, g_loss, d_loss[0], d_loss[1]))
            if (i + 1) % display_step == 0:
                self.show_image(save_path.format(i+1),\
                    i+1, plot_shape)
        return loss_dict

#%%
