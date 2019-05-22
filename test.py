#%%
from keras.layers import Dense, Input, Dropout, Flatten, Reshape, BatchNormalization
from keras.models import Model, Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
#%%
input_shape = 784
z_shape = 100
z = Input(shape = (z_shape,))
epochs = 100
batch_size = 600
lr = 0.0002
def get_discriminator():
    _input = Input(shape = (input_shape,))
    x = Dense(128)(_input)
    x = LeakyReLU(0.2)(x)
    #x = Dropout(0.5)(x)
    x = Dense(64)(x)
    x = LeakyReLU(0.2)(x)
    x = Dense(32)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation = 'sigmoid')(x)

    model = Model(_input, x)
    model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

def get_generator(_input):
    _input = Input(shape = (z_shape,))
    x = Dense(64)(_input)
    x = LeakyReLU(0.2)(x)
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum = 0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization(momentum = 0.8)(x)
    #x = Dropout(0.5)(x)
    x = Dense(784, activation = 'sigmoid')(x)

    model = Model(_input, x)
    return model
    
def show_image(n = 10):
        
    noise = np.random.normal(0, 1, (n, z_shape))
    imgs = generator.predict(noise)
    #imgs = imgs * 127.5 + 127.5
    imgs = imgs * 255
    #imgs = np.clip(imgs, 0, 255).astype(np.uint8)

    _, axes = plt.subplots(2, 5, figsize = (10,2))
    axes = axes.flatten()

    for (img, ax) in zip(imgs, axes):
        img = img.reshape((28, 28))
        #print(img.shape)
        ax.imshow(img, cmap = 'gray')
    plt.show()

discriminator =  get_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=Adam(lr, 0.5),
    metrics=['accuracy'])
#%%
generator = get_generator(z)
discriminator.trainable = False
gan_input = Input(shape = (z_shape,))
img = generator(gan_input)
x = discriminator(img)

adversarial = Model(gan_input, x)
adversarial.compile(loss = 'binary_crossentropy',optimizer = Adam(lr, 0.5))
#%%
(x_train, _), (_, _) = mnist.load_data()
#x_train = (x_train - 127.5) / 127.5
x_train = x_train / 255.0
#x_train = np.expand_dims(x_train, axis=3)
x_train = np.reshape(x_train, (x_train.shape[0], input_shape))
n_data = len(x_train)
loss_dict = {'d_loss':[],'g_loss':[]}
for i in np.arange(epochs):
    print ('{} epochs train start'.format(i+1))
    for _ in np.arange(0, n_data, batch_size):
        idx = np.random.randint(0, high = n_data, size = batch_size)
        real_x = x_train[idx]
        real_y = np.ones((batch_size, 1))
        #real_y[:] = 0.9
        # train discriminator
        #discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_x, real_y)
        #for _ in np.arange(20):
        z = np.random.normal(0, 1, (batch_size, z_shape))
        fake = generator.predict_on_batch(z)
        fake_y = np.zeros((batch_size, 1))

        d_loss_fake = discriminator.train_on_batch(fake, fake_y)
        
        # train generator
        #discriminator.trainable = False
        z = np.random.normal(0, 1, (batch_size, z_shape))
        g_loss = adversarial.train_on_batch(z, real_y)
        d_loss = np.add(d_loss_real, d_loss_fake) * 0.5
        loss_dict['g_loss'].append(g_loss)
        loss_dict['d_loss'].append(d_loss[0])
    print('generator loss :{:.4f} / discriminator loss :{:.4f}'.format(g_loss, d_loss[0]))    
    if (i + 1) % 10 == 0:
        show_image(n = 25)


plt.style.use('seaborn')
plt.plot(range(int(epochs * n_data / batch_size)), loss_dict['d_loss'], label = 'd_loss', )
plt.plot(range(int(epochs * n_data / batch_size)), loss_dict['g_loss'], label = 'g_loss')
plt.legend()
#plt.savefig('./example.png')
plt.show()
