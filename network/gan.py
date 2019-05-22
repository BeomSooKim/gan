from keras.layers import Dense, Input, Dropout
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist

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
        self.discriminator.trainable = False
        fake_img = self.discriminator(img)

        self.adversarial = Model(z, fake_img)
        self.adversarial.compile(loss = 'binary_crossentropy',optimizer = self.optimizer)


    def get_discriminator(self):
        _input = Input(shape = (self.input_shape,))
        x = Dense(128)(_input)
        x = LeakyReLU(0.3)(x)
        #x = Dropout(0.5)(x)
        x = Dense(64)(x)
        x = LeakyReLU(0.3)(x)
        x = BatchNormalization()(x)
        x = Dense(32)(x)
        x = LeakyReLU(0.3)(x)
        x = Dropout(rate = 0.5)(x)
        x = Dense(1, activation = 'sigmoid')(x)

        model = Model(_input, x)
        model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model

    def get_generator(self, _input):
        _input = Input(shape = (self.z_shape,))
        x = Dense(64)(_input)
        x = LeakyReLU(0.3)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dense(128)(x)
        x = LeakyReLU(0.3)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dense(256)(x)
        x = LeakyReLU(0.3)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.3)(x)
        x = BatchNormalization(momentum = 0.8)(x)
        #x = Dropout(0.5)(x)
        x = Dense(784, activation = 'tanh')(x)

        model = Model(_input, x)
        return model

    def show_image(self, savepath, shape = (2, 5)):
        
        noise = np.random.normal(0, 1, (np.prod(shape), self.z_shape))
        imgs = self.generator.predict(noise)
        imgs = imgs * 127.5 +127.5
        imgs = imgs.reshape((np.prod(shape), 28, 28))

        fig, axes = plt.subplots(nrows = shape[0],ncols = shape[1], figsize = np.array(shape)[::-1]*2)
        axes = axes.flatten()

        for (img, ax) in zip(imgs, axes):
            ax.imshow(img, cmap = 'gray')
        fig.savefig(savepath, facecolor = 'w')


    def train(self, epochs, batch_size, display_step, plot_shape):
        (x_train, _), (_, _) = mnist.load_data()
        x_train = (x_train - 127.5) / 127.5
        x_train = np.reshape(x_train, (x_train.shape[0], self.input_shape))
        n_data = len(x_train)
        loss_dict = {"d_loss":[],"d_acc":[], "g_loss":[]}
        for i in np.arange(epochs):
            for _ in np.arange(0, n_data, batch_size):
                idx = np.random.randint(0, high = n_data, size = batch_size)
                # train discriminator
                real_x = x_train[idx]
                real_y = np.ones((batch_size, 1)) - 0.05 *np.random.uniform(0, 1, (batch_size, 1))
                
                z = np.random.normal(0, 1, (batch_size, self.z_shape))
                fake_x = self.generator.predict_on_batch(z)
                fake_y = np.zeros((batch_size, 1)) + 0.05 *np.random.uniform(0, 1, (batch_size, 1))

                #all_x = np.concatenate((real_x, fake_x))
                #all_y = np.concatenate((real_y, fake_y))
                #d_loss = self.discriminator.train_on_batch(all_x, all_y)
                d_loss1 = self.discriminator.train_on_batch(real_x, real_y)
                d_loss2 = self.discriminator.train_on_batch(fake_x, fake_y)
                d_loss = np.add(d_loss1, d_loss2) * 0.5
                #train generator
                z = np.random.normal(0, 1, (batch_size, self.z_shape))
                g_loss = self.adversarial.train_on_batch(z, real_y)

                loss_dict['d_loss'].append(d_loss[0])
                loss_dict['d_acc'].append(d_loss[1])
                loss_dict['g_loss'].append(g_loss)
                
            print('{} epoch status :gen_loss = {:.4f} / disc_loss = {:.4f}'\
               .format(i+1, g_loss, d_loss[0]))
            if (i + 1) % display_step == 0:
                self.show_image("D:\\study\\4.experiments\\GAN\\images\\{}epoch.png".format(i+1), plot_shape)
        return loss_dict


    


