from keras.layers import Dense, Input, Dropout
from keras.layers import LeakyReLU, BatchNormalization, ReLU
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist
from keras import initializers

import matplotlib.pyplot as plt

class GAN:
    def __init__(self, input_shape, z_shape, optimizer):
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.optimizer = optimizer

        self.discriminator = self.get_discriminator()

        z = Input(shape = (self.z_shape,))
        self.generator = self.get_generator(z)
        
        img = self.generator(z)
        self.discriminator.trainable = False #for training G, D weight must be fixed
        fake_img = self.discriminator(img)

        self.adversarial = Model(z, fake_img)
        self.adversarial.compile(loss = 'binary_crossentropy',optimizer = self.optimizer)

    # build discriminator D
    def get_discriminator(self):
        _input = Input(shape = (self.input_shape,))
        x = Dense(1024)(_input)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(512)(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation = 'sigmoid')(x) # output is between 0 and 1 

        model = Model(_input, x)
        model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model
    # build generator G
    def get_generator(self, _input):
        _input = Input(shape = (self.z_shape,))
        x = Dense(256)(_input)
        #x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(512)(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1024)(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(784, activation = 'tanh')(x) # for MNIST image size

        model = Model(_input, x)
        return model

    def show_image(self, savepath, shape = (2, 5)):
        
        noise = np.random.normal(0, 1, (np.prod(shape), self.z_shape))
        imgs = self.generator.predict(noise)
        # decode image (0 ~ 255)
        imgs = imgs * 127.5 +127.5
        imgs = np.clip(imgs, 0, 255).astype(np.uint8)
        imgs = imgs.reshape((np.prod(shape), 28, 28))

        fig, axes = plt.subplots(nrows = shape[0],ncols = shape[1], figsize = np.array(shape)[::-1]*2)
        axes = axes.flatten()

        for (img, ax) in zip(imgs, axes):
            ax.imshow(img, cmap = 'gray')
        fig.savefig(savepath, facecolor = 'w')


    def train(self, epochs, batch_size, display_step, plot_shape, save_path):
        (x_train, _), (_, _) = mnist.load_data()
        # encode image
        x_train = (x_train - 127.5) / 127.5
        x_train = np.reshape(x_train, (x_train.shape[0], self.input_shape))
        n_data = len(x_train)
        loss_dict = {"d_loss":[],"d_acc":[], "g_loss":[]}
        for i in np.arange(epochs):
            for _ in np.arange(0, n_data, batch_size):
                idx = np.random.randint(0, high = n_data, size = batch_size)
                # train discriminator
                real_x = x_train[idx]
                z = np.random.normal(0, 1, (batch_size, self.z_shape))
                fake_x = self.generator.predict_on_batch(z)
                all_x = np.concatenate((real_x, fake_x))
                all_y = np.zeros(2*batch_size)
                all_y[:batch_size] = 1 #for gradient, set real label to 0.9
                
                d_loss = self.discriminator.train_on_batch(real_x, real_y)
                d_loss = self.discriminator.train_on_batch(fake_x, fake_y)
                
                #train generator
                z = np.random.normal(0, 1, (batch_size, self.z_shape))
                g_loss = self.adversarial.train_on_batch(z, np.ones((batch_size, 1)))

                loss_dict['d_loss'].append(d_loss[0])
                loss_dict['d_acc'].append(d_loss[1])
                loss_dict['g_loss'].append(g_loss)
                
            print('{} epoch status :gen_loss = {:.4f} / disc_loss = {:.4f}'\
               .format(i+1, g_loss, d_loss[0]))
            if (i + 1) % display_step == 0:
                self.show_image(save_path.format(i+1), plot_shape)
        return loss_dict


    


