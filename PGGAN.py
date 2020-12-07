# Progressive Growing GAN on celebrity faces dataset
import tensorflow as tf
from tensorflow.keras import layers, models, initializers, constraints, optimizers
import tensorflow.keras.backend as K
import numpy as np

import os
from os.path import isfile, isdir, join

from skimage import io
from skimage.transform import resize
from matplotlib import pyplot

# pixel-wise feature vector normalization layer
class PixelNormalization(layers.Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # L2 norm for all the activations in the same image and at the same location across all channels
    def call(self, inputs):
        return inputs / K.sqrt(K.mean(inputs**2, axis=-1, keepdims=True) + 1e-8)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape

# mini-batch standard deviation layer
class MinibatchStdev(layers.Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # calculate the mean standard deviation across each pixel coord
    def call(self, inputs):
        mean = K.mean(inputs, axis=0, keepdims=True)
        mean_sq_diff = K.mean(K.square(inputs - mean), axis=0, keepdims=True) + 1e-8
        mean_pix = K.mean(K.sqrt(mean_sq_diff), keepdims=True)
        shape = K.shape(inputs)
        output = K.tile(mean_pix, [shape[0], shape[1], shape[2], 1])
        return K.concatenate([inputs, output], axis=-1)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        input_shape = list(input_shape)
        input_shape[-1] += 1
        return tuple(input_shape)

# weighted sum output
class WeightedSum(layers.Add):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = K.variable(0., name='ws_alpha')

    # output a weighted sum of inputs, only supports a weighted sum of two inputs
    def _merge_function(self, inputs):
        assert (len(inputs) == 2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def add_discriminator_block(old_model, n_input_layers=3):
    init = initializers.RandomNormal(stddev=0.02)
    const = constraints.max_norm(1.0)
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    # define new input shape as double the size
    input_shape = (in_shape[-3]*2, in_shape[-2]*2, in_shape[-1])
    in_image = layers.Input(shape=input_shape)
    # define new input processing layer
    d = layers.Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # define new block
    d = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    d = layers.AveragePooling2D()(d)
    block_new = d
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model1 = models.Model(in_image, d)
    model1.compile(loss=wasserstein_loss, optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    # downsample the new larger image
    downsample = layers.AveragePooling2D()(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old, block_new])
    # skip the input, 1x1 and activation for the old model
    for i in range(n_input_layers, len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model2 = models.Model(in_image, d)
    model2.compile(loss=wasserstein_loss, optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    return [model1, model2]

# define the discriminator models for each image resolution
def define_discriminator(n_blocks, input_shape=(4,4,3)):
    init = initializers.RandomNormal(stddev=0.02)
    const = constraints.max_norm(1.0)
    model_list = list()
    in_image = layers.Input(shape=input_shape)
    # conv 1x1
    d = layers.Conv2D(128, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(in_image)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # conv 3x3 (output block)
    d = MinibatchStdev()(d)
    d = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # conv 4x4
    d = layers.Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(d)
    d = layers.LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = layers.Flatten()(d)
    out_class = layers.Dense(1)(d)

    model = models.Model(in_image, out_class)
    model.compile(loss=wasserstein_loss, optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i-1][0]
        # create new model for next resolution
        new_models = add_discriminator_block(old_model)
        model_list.append(new_models)
    return model_list

def add_generator_block(old_model):
    init = initializers.RandomNormal(stddev=0.02)
    const = constraints.max_norm(1.0)
    # get the end of the last block
    block_end = old_model.layers[-2].output
    # upsample, and define new block
    upsampling = layers.UpSampling2D()(block_end)
    g = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    g = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # add new output layer
    out_image = layers.Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

    model1 = models.Model(old_model.input, out_image)
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
    # define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])

    model2 = models.Model(old_model.input, merged)
    return [model1, model2]

def define_generator(latent_dim, n_blocks, in_dim=4):
    init = initializers.RandomNormal(stddev=0.02)
    const = constraints.max_norm(1.0)
    model_list = list()
    in_latent = layers.Input(shape=(latent_dim,))
    # linear scale up to activation maps
    g  = layers.Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = layers.Reshape((in_dim, in_dim, 128))(g)
    # conv 4x4, input block
    g = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # conv 3x3
    g = layers.Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = layers.LeakyReLU(alpha=0.2)(g)
    # conv 1x1, output block
    out_image = layers.Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)

    model = models.Model(in_latent, out_image)
    model_list.append([model, model])
    # create submodels
    for i in range(1, n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i-1][0]
        # create new model for next resolution
        new_models = add_generator_block(old_model)
        model_list.append(new_models)
    return model_list

# define composite models for training generators via discriminators
def define_composite(discriminators, generators):
    model_list = list()
    for i in range(len(discriminators)):
        g_models, d_models = generators[i], discriminators[i]
        # straight-through model
        d_models[0].trainable = False
        model1 = models.Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(loss=wasserstein_loss, optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # fade-in model
        d_models[1].trainable = False
        model2 = models.Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(loss=wasserstein_loss, optimizer=optimizers.Adam(lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        model_list.append([model1, model2])
    return model_list

# load images from folder
def load_real_samples(imgs_path):
    imgs = []
    for img_name in os.listdir(imgs_path):
        img_path = join(imgs_path, img_name)
        if isfile(img_path):
            img = io.imread(img_path)
            img = np.float32(img)
            imgs.append(img)
    imgs = np.array(imgs, dtype=np.float32)
    # scale from [0,255] to [-1,1]
    imgs = imgs / 127.5 - 1.
    return imgs

# select real samples
def generate_real_samples(dataset, n_samples):
    # select random instances
    X = dataset[np.random.randint(0, dataset.shape[0], n_samples)]
    # generate class labels
    y = np.ones((n_samples, 1))
    return X, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = np.random.randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -np.ones((n_samples, 1))
    return X, y

# update the alpha value on each instance of WeightedSum
def update_fadein(models, step, n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer, WeightedSum):
                K.set_value(layer.alpha, alpha)

# train a generator and discriminator
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_steps):
        # update alpha for all WeightedSum layers when fading in new blocks
        if fadein:
            update_fadein([g_model, d_model, gan_model], i, n_steps)
        # prepare real and fake samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update the generator via the discriminator's error
        z_input = generate_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))

# scale images to preferred size with nearest neighbor interpolation
def scale_dataset(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)

# generate samples and save as a plot and save the model
def summarize_performance(status, g_model, latent_dim, n_samples=25):
    gen_shape = g_model.output_shape
    name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)

    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # normalize pixel values to the range [0,1]
    X = (X - X.min()) / (X.max() - X.min())
    # plot real images
    square = int(np.sqrt(n_samples))
    for i in range(n_samples):
        pyplot.subplot(square, square, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X[i])
    # save plot to file
    filename1 = 'plot_%s.png' % (name)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%s.h5' % (name)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# train the generator and discriminator
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    # fit the baseline model
    g_normal, d_normal, gan_normal = g_models[0][0], d_models[0][0], gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance('tuned', g_normal, latent_dim)
    # process each level of growth
    for i in range(1, len(g_models)):
        # retrieve models for this level of growth
        [g_normal, g_fadein] = g_models[i]
        [d_normal, d_fadein] = d_models[i]
        [gan_normal, gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print('Scaled Data', scaled_data.shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i], True)
        summarize_performance('faded', g_fadein, latent_dim)
        # train normal or straight-through models
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance('tuned', g_normal, latent_dim)

# call this function when using GPU with small memory
def using_gpu_memory_growth():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    return config

if __name__ == "__main__":
    using_gpu_memory_growth()
    # number of growth phases, e.g. 6 == [4, 8, 16, 32, 64, 128]
    n_blocks = 6
    # size of the latent space
    latent_dim = 100

    n_batch = [16, 16, 16, 8, 4, 4]
    # 1 epochs == 3K images per training phase
    n_epochs = [5, 8, 8, 10, 10, 10]

    dataset = load_real_samples('./Projects/GAN_Exp/Face_64X64/')
    print('Loaded', dataset.shape)

    d_models = define_discriminator(n_blocks)
    g_models = define_generator(latent_dim, n_blocks)
    gan_models = define_composite(d_models, g_models)

    train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)