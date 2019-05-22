#%%
from keras.layers import Dense, Input, Dropout, Reshape, ReLU, Flatten
from keras.layers import LeakyReLU, BatchNormalization, Conv2D, Conv2DTranspose, Activation
from keras.layers import MaxPooling2D, Concatenate
from keras.models import Model
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist

import matplotlib.pyplot as plt
#%%
class CDCGAN:
    def __init__(self, input_shape, z_shape, condition_shape, optimizer):
        self.input_shape = input_shape
        self.z_shape = z_shape
        self.condition = condition_shape
        self.optimizer = optimizer

        self.discriminator = self.get_discriminator()

        z = Input(shape = (self.z_shape,))
        c_input = Input(shape = (self.condition,))
        self.generator = self.get_generator(z, c_input)
        
        img = self.generator([z, c_input])
        self.discriminator.trainable = False
        fake_img = self.discriminator([img, c_input])

        self.adversarial = Model([z, c_input], fake_img)
        self.adversarial.compile(loss = 'binary_crossentropy',optimizer = self.optimizer)


    def get_discriminator(self):
        _input = Input(shape = self.input_shape, name = 'x_input')
        c_input = Input(shape = (self.condition,), name = 'c_input')
        x = Conv2D(filters = 32, kernel_size = (3,3), strides = (2,2), padding = 'same')(_input)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        
        x = Conv2D(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        
        x = Conv2D(filters = 128, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        x = Flatten()(x)
        x = Concatenate()([x, c_input])
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        x = Dense(1, activation = 'sigmoid')(x)

        model = Model([_input, c_input], x)
        model.compile(optimizer = self.optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
        
        return model

    def get_generator(self, _input, c_input):
        
        _input = Input(shape = (self.z_shape,), name = 'x_input')
        c_input = Input(shape = (self.condition,), name = 'c_input')
        x = Concatenate()([_input, c_input])
        x = Dense(7*7*128)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Reshape((7, 7, 128))(x)
        x = Conv2DTranspose(filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2DTranspose(filters = 32, kernel_size = (3, 3), strides = (2,2), padding = 'same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters = 1, kernel_size = (3,3),padding = 'same', activation = 'tanh')(x)

        model = Model([_input, c_input], x)
        return model

    def show_image(self, savepath, shape = (2, 5)):
        np.random.seed(13245)
        noise = np.random.normal(0, 0.5, (np.prod(shape), self.z_shape))
        condition = np.random.randint(0, 10, np.prod(shape))
        condition_encode = to_categorical(condition, num_classes = 10)
        imgs = self.generator.predict([noise, condition_encode])
        imgs = imgs * 127.5 +127.5
        #imgs = imgs.reshape((np.prod(shape), 28, 28))

        fig, axes = plt.subplots(nrows = shape[0],ncols = shape[1], figsize = np.array(shape)[::-1]*2)
        axes = axes.flatten()

        for (img, c, ax) in zip(imgs, condition, axes):
            ax.imshow(np.squeeze(img), cmap = 'gray')
            ax.set_title(c)

        #plt.show()
        fig.savefig(savepath, facecolor = 'w')
        plt.close()
    def train(self, epochs, batch_size, display_step, plot_shape):
        (x_train, y_train), (_, _) = mnist.load_data()
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = x_train.reshape(*x_train.shape, 1)

        y_train = to_categorical(y_train, num_classes= 10)
        n_data = len(x_train)
        loss_dict = {"d_loss":[], "g_loss":[]}
        for i in np.arange(epochs):
            for _ in np.arange(0, n_data, batch_size):
                idx = np.random.randint(0, high = n_data, size = batch_size)
                # train discriminator
                real_x = x_train[idx]
                real_cx = y_train[idx]
                real_y = np.ones((batch_size, 1)) - 0.05 *np.random.uniform(0, 1, (batch_size, 1))
                
                z = np.random.uniform(-1, 1, (batch_size, self.z_shape))
                fake_x = self.generator.predict([z, real_cx])
                fake_y = np.zeros((batch_size, 1)) + 0.05 *np.random.uniform(0, 1, (batch_size, 1))

                #all_x = np.concatenate((real_x, fake_x))
                #all_y = np.concatenate((real_y, fake_y))
                d_loss1 = self.discriminator.train_on_batch([real_x, real_cx], real_y)
                d_loss2 = self.discriminator.train_on_batch([fake_x, real_cx], fake_y)
                d_loss = (np.array(d_loss1) +np.array(d_loss2))/2
                #train generator
                #z = np.random.normal(-1, 1, (batch_size, self.z_shape))
                g_loss = self.adversarial.train_on_batch([z, real_cx], real_y)

                loss_dict['d_loss'].append(d_loss[0])
                #loss_dict['d_acc'].append(d_loss[1])
                loss_dict['g_loss'].append(g_loss)
                
            print('{} epoch status :gen_loss = {:.4f} / disc_loss = {:.4f} / disc_acc = {:.4f}'\
               .format(i+1, g_loss, d_loss[0], d_loss[1]))
            if (i + 1) % display_step == 0:
                self.show_image("D:\\study\\4.experiments\\GAN\\images\\cdcgan\\{}epoch.png".format(i+1),plot_shape)
        return loss_dict

#%%
