'''
    Multi-Scale Gradients GAN on celebrity faces dataset
    CelebA dataset: https://www.kaggle.com/jessicali9530/celeba-dataset
'''
import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
import random

import os
from os.path import isfile, isdir, join
import time
import datetime

import cv2 as cv
import imutils

save_every_number_episodes = 50
max_checkpoints_number = 3
log_dir = "./Projects/GAN_Exp/logs/"
saved_model_path = "./Projects/GAN_Exp/saved_models/"
images_path = "./Projects/GAN_Exp/Face_64X64/"
load_model_id = ""

EPOCHS = 100
BATCH_SIZE = 12
LATENT_DIM = 256

IS_MSG = True

lr_g_max = 3e-4
lr_g_min = 1e-4
beta_1_g = 0.
beta_2_g = 0.9
lr_d_max = 3e-4
lr_d_min = 1e-4
beta_1_d = 0.
beta_2_d = 0.9

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

class MSGGAN:
    def __init__(self, latent_dim, beta_1_g, beta_2_g, beta_1_d, beta_2_d, is_msg):
        self.latent_dim = latent_dim
        self.is_msg = is_msg

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()

        self.generator_optimizer = keras.optimizers.Adam(beta_1=beta_1_g, beta_2=beta_2_g)
        self.discriminator_optimizer = keras.optimizers.Adam(beta_1_d, beta_2_d)
        #self.generator_optimizer = keras.optimizers.RMSprop()
        #self.discriminator_optimizer = keras.optimizers.RMSprop()

    def stack_layer(self, filters, size, strides=1, padding='same', k_init_stddev=1., max_norm=0., power_iterations=-1, 
                        norm_func=None, dropout_rate=0., conv_func=layers.Conv2D, activation=tf.nn.leaky_relu):
        result = keras.Sequential()

        kernel_initializer = keras.initializers.RandomNormal(stddev=k_init_stddev) if k_init_stddev > 0 else 'glorot_uniform'
        kernel_constraint = keras.constraints.MaxNorm(max_norm) if max_norm > 0 else None
        conv = conv_func(filters, size, strides=strides,
                                        padding=padding,
                                        kernel_initializer=kernel_initializer, 
                                        kernel_constraint=kernel_constraint)
        if power_iterations > 0:
            conv = SpectralNormalization(conv, power_iterations=power_iterations)
        result.add(conv)

        if norm_func is not None:
            result.add(norm_func())

        if dropout_rate > 0:
            result.add(layers.Dropout(dropout_rate))

        if activation is not None:
            result.add(layers.Activation(activation))

        return result

    def dense_to_conv(self, x, conv_shape, k_init_stddev=1., max_norm=0.):
        kernel_initializer = keras.initializers.RandomNormal(stddev=k_init_stddev) if k_init_stddev > 0 else 'glorot_uniform'
        kernel_constraint = keras.constraints.MaxNorm(max_norm) if max_norm > 0 else None
        x = layers.Dense(tf.reduce_prod(conv_shape).numpy(), 
                            kernel_initializer=kernel_initializer, 
                            kernel_constraint=kernel_constraint)(x)
        x = layers.Reshape(conv_shape)(x)
        return x

    def make_generator_model(self):
        outputs = []
        latent = layers.Input(shape=(self.latent_dim, ))
        x = self.dense_to_conv(latent, (4, 4, 512))     # (bs, 4, 4, 512)
        '''
        x = layers.Reshape((1, 1, self.latent_dim))(latent)          # (bs, 1, 1, latent_dim)
        x = self.stack_layer(512, 4, strides=4, conv_func=layers.Conv2DTranspose)(x)   # (bs, 4, 4, 512)
        '''
        x = self.stack_layer(512, 4, norm_func=PixelNormalization)(x)
        if self.is_msg:
            o = self.stack_layer(3, 1, activation=None)(x)    # (bs, 4, 4, 3)
            outputs.append(o)

        x = layers.UpSampling2D(2)(x)    # (bs, 8, 8, 512)
        x = self.stack_layer(512, 4, norm_func=PixelNormalization)(x)   
        x = self.stack_layer(512, 4, norm_func=PixelNormalization)(x)
        if self.is_msg:
            o = self.stack_layer(3, 1, activation=None)(x)    # (bs, 8, 8, 3)
            outputs.append(o)

        x = layers.UpSampling2D(2)(x)    # (bs, 16, 16, 512)
        x = self.stack_layer(512, 4, norm_func=PixelNormalization)(x)   
        x = self.stack_layer(512, 4, norm_func=PixelNormalization)(x)
        if self.is_msg:
            o = self.stack_layer(3, 1, activation=None)(x)    # (bs, 16, 16, 3)
            outputs.append(o)

        x = layers.UpSampling2D(2)(x)    # (bs, 32, 32, 512)
        x = self.stack_layer(512, 5, norm_func=PixelNormalization)(x)   
        x = self.stack_layer(512, 5, norm_func=PixelNormalization)(x)
        if self.is_msg:
            o = self.stack_layer(3, 1, activation=None)(x)    # (bs, 32, 32, 3)
            outputs.append(o)

        x = layers.UpSampling2D(2)(x)    # (bs, 64, 64, 512)
        x = self.stack_layer(256, 5, norm_func=PixelNormalization)(x)  # (bs, 64, 64, 256)
        x = self.stack_layer(256, 5, norm_func=PixelNormalization)(x)
        o = self.stack_layer(3, 1, activation=None)(x)    # (bs, 64, 64, 3)
        outputs.append(o)
        '''
        x = layers.UpSampling2D(2)(x)    # (bs, 128, 128, 256)
        x = self.stack_layer(128, 3, norm_func=PixelNormalization)(x)  # (bs, 128, 128, 128)
        x = self.stack_layer(128, 3, norm_func=PixelNormalization)(x)
        o = self.stack_layer(3, 1)(x)    # (bs, 128, 128, 3)
        outputs.append(o)
        '''
        
        return keras.Model(inputs=latent, outputs=outputs)

    def make_discriminator_model(self):
        inputs = []
        '''
        i = layers.Input(shape=(128, 128, 3))
        inputs.append(i)
        x = MinibatchStdev()(i)
        x = self.stack_layer(128, 3)(x)  # (bs, 128, 128, 128)
        x = self.stack_layer(256, 3)(x)  # (bs, 128, 128, 256)
        x = layers.AvgPool2D(2)(x)       # (bs, 64, 64, 256)
        '''
        i = layers.Input(shape=(64, 64, 3))
        inputs.append(i)
        #x = layers.Concatenate()([i, x])
        x = MinibatchStdev()(i)
        x = self.stack_layer(256, 5, power_iterations=5)(x)  # (bs, 64, 64, 256)
        x = self.stack_layer(512, 5, power_iterations=5)(x)  # (bs, 64, 64, 512)
        x = layers.AvgPool2D(2)(x)       # (bs, 32, 32, 512)

        if self.is_msg:
            i = layers.Input(shape=(32, 32, 3))
            inputs.append(i)
            x = layers.Concatenate()([i, x])
            x = MinibatchStdev()(x)
        x = self.stack_layer(512, 5, power_iterations=5)(x)  # (bs, 32, 32, 512)
        x = self.stack_layer(512, 5, power_iterations=5)(x)  # (bs, 32, 32, 512)
        x = layers.AvgPool2D(2)(x)       # (bs, 16, 16, 512)

        if self.is_msg:
            i = layers.Input(shape=(16, 16, 3))
            inputs.append(i)
            x = layers.Concatenate()([i, x])
            x = MinibatchStdev()(x)
        x = self.stack_layer(512, 4, power_iterations=5)(x)  # (bs, 16, 16, 512)
        x = self.stack_layer(512, 4, power_iterations=5)(x)  # (bs, 16, 16, 512)
        x = layers.AvgPool2D(2)(x)       # (bs, 8, 8, 512)

        if self.is_msg:
            i = layers.Input(shape=(8, 8, 3))
            inputs.append(i)
            x = layers.Concatenate()([i, x])
            x = MinibatchStdev()(x)
        x = self.stack_layer(512, 3, power_iterations=5)(x)  # (bs, 8, 8, 512)
        x = self.stack_layer(512, 3, power_iterations=5)(x)  # (bs, 8, 8, 512)
        x = layers.AvgPool2D(2)(x)       # (bs, 4, 4, 512)

        if self.is_msg:
            i = layers.Input(shape=(4, 4, 3))
            inputs.append(i)
            x = layers.Concatenate()([i, x])
            x = MinibatchStdev()(x)
        x = self.stack_layer(512, 3, power_iterations=5)(x)  # (bs, 4, 4, 512)
        x = self.stack_layer(512, 4, power_iterations=5, padding='valid')(x)  # (bs, 1, 1, 512)
        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)          # (bs, 1, 1, 1)

        return keras.Model(inputs=inputs, outputs=x)

    def sigmoid_cross_entropy_loss(self, target, output):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.constant(target, dtype=tf.float32, shape=output.shape), output))

    def wasserstein_loss(self, target, output):
        return -1. * tf.reduce_mean(target * output)

    def hinge_loss(self, target, output):
        return -1. * tf.reduce_mean(tf.minimum(0., -1. + target * output))

    @tf.function
    def train_step(self, real_images, batch_size, lr_g, lr_d):
        latent = self.generate_latent(batch_size)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_outputs = self.generator(latent, training=True)
            if self.is_msg:
                gen_outputs.reverse()

            disc_real_output = self.discriminator(real_images, training=True)
            disc_generated_output = self.discriminator(gen_outputs, training=True)

            #disc_loss_real = self.sigmoid_cross_entropy_loss(.9, disc_real_output)
            #disc_loss_gen = self.sigmoid_cross_entropy_loss(0., disc_generated_output)
            #disc_loss_real = self.wasserstein_loss(1., disc_real_output)
            #disc_loss_gen = self.wasserstein_loss(-1., disc_generated_output)
            disc_loss_real = self.hinge_loss(1., disc_real_output)
            disc_loss_gen = self.hinge_loss(-1., disc_generated_output)
            disc_total_loss = disc_loss_real + disc_loss_gen

            #gen_total_loss = self.sigmoid_cross_entropy_loss(.9, disc_generated_output)
            gen_total_loss = self.wasserstein_loss(1., disc_generated_output)

        self.discriminator_optimizer.learning_rate = lr_d
        self.generator_optimizer.learning_rate = lr_g
        discriminator_gradients = disc_tape.gradient(disc_total_loss, self.discriminator.trainable_variables)
        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

        train_dict = {"disc_total_loss": disc_total_loss, "gen_total_loss": gen_total_loss, 
                    "max disc_grads[-1]": tf.reduce_max(discriminator_gradients[-1]), "max disc_grads[0]": tf.reduce_max(discriminator_gradients[0]), 
                    "max gen_grads[-1]": tf.reduce_max(generator_gradients[-1]), "max gen_grads[0]": tf.reduce_max(generator_gradients[0])}
        return train_dict

    def generate_latent(self, batch_size=1):
        latent = tf.random.normal([batch_size, self.latent_dim])
        #latent = tf.math.l2_normalize(latent, axis=1)
        return latent

    def generate_samples(self, batch_size=1):
        return self.generator(self.generate_latent(batch_size), training=True)

def load_model(path, model, max_checkpoints_number):
    if path is not None:
        tf.print("Loading model...from: {}".format(path))
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=max_checkpoints_number)
        ckpt.restore(manager.latest_checkpoint)

def save_model(path, model, max_checkpoints_number):
    if path is not None:
        tf.print("Saveing model...from: {}".format(path))
        ckpt = tf.train.Checkpoint(model=model)
        manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=max_checkpoints_number)
        manager.save()

def load_images(imgs_path):
    imgs = []
    for img_name in os.listdir(imgs_path):
        img_path = join(imgs_path, img_name)
        if isfile(img_path):
            img = cv.imread(img_path)
            img = np.float32(img)
            imgs.append(img)
    return imgs

def image_normalization(img, img_min=0, img_max=255):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)
    :param img: an image could be gray scale or color
    :param img_min: for default is 0
    :param img_max: for default is 255
    :return: a normalized image, if max is 255 the dtype is uint8
    """
    img = np.float32(img)
    epsilon=1e-12 # whenever an inconsistent image
    img = (img-np.min(img))*(img_max-img_min)/((np.max(img)-np.min(img))+epsilon)+img_min
    return img

def generate_and_save_samples_l(gan, W=64, H=64, save_path="./", name="sample"):
    outputs = gan.generate_samples()
    concat_images = []
    if gan.is_msg:
        for img in outputs:
            img = image_normalization(img)[0, :]
            img = cv.resize(img, (W, H), interpolation=cv.INTER_NEAREST)
            concat_images.append(img)
        concat_output = cv.hconcat(concat_images)
    else:
        concat_output = image_normalization(outputs)[0, :]
    cv.imwrite(save_path + name + ".png", concat_output)

def resize_batch_images(imgs, img_min=-1., img_max=1., size_list=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]):
    batch_resized_images =[]
    for W, H in size_list:
        resized_list = []
        for img in imgs:
            img = image_normalization(img, img_min=img_min, img_max=img_max)
            resized_img = cv.resize(img, (W, H), interpolation=cv.INTER_AREA)            
            resized_list.append(resized_img)
        batch_resized_images.append(np.array(resized_list))
    return batch_resized_images

def data_augmentation(img, min_rot_angle=-180, max_rot_angle=180, crop_ratio=0.2, smooth_size=3, sharp_val=3, max_noise_scale=10):
    (H, W) = img.shape[:2]
    img_a = img

    all_func = ['flip', 'rotate', 'crop', 'smooth', 'sharp', 'noise']
    do_func = np.random.choice(all_func, size=np.random.randint(1, len(all_func)), replace=False)
    #do_func = ['crop']
    # Filp image, 0: vertically, 1: horizontally
    if 'flip' in do_func:
        img_a = cv.flip(img_a, np.random.choice([0, 1]))
    # Rotate image
    if 'rotate' in do_func:
        rot_ang = np.random.uniform(min_rot_angle, max_rot_angle)
        img_a = imutils.rotate_bound(img_a, rot_ang)
    # Crop image
    if 'crop' in do_func:
        (H_A, W_A) = img_a.shape[:2]
        start_x = np.random.randint(0, int(H_A * crop_ratio))
        start_y = np.random.randint(0, int(W_A * crop_ratio))
        end_x = np.random.randint(int(H_A * (1-crop_ratio)), H_A)
        end_y = np.random.randint(int(W_A * (1-crop_ratio)), W_A)

        img_a = img_a[start_x:end_x, start_y:end_y]
    # Smoothing
    if 'smooth' in do_func:
        img_a = cv.GaussianBlur(img_a, (smooth_size, smooth_size), 0)
    # Sharpening
    if 'sharp' in do_func:
        de_sharp_val = -(sharp_val - 1) / 8
        kernel = np.array([[de_sharp_val]*3, [de_sharp_val, sharp_val, de_sharp_val], [de_sharp_val]*3])
        img_a = cv.filter2D(img_a, -1, kernel)
    # Add the Gaussian noise to the image
    if 'noise' in do_func:
        noise_scale = np.random.uniform(0, max_noise_scale)
        gauss = np.random.normal(0, noise_scale, img_a.size)
        gauss = np.float32(gauss.reshape(img_a.shape[0],img_a.shape[1],img_a.shape[2]))
        img_a = cv.add(img_a,gauss)
    # Keep shape
    img_a = cv.resize(img_a, (W, H))
    return np.float32(img_a)

def train():
    ds = load_images(images_path)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + current_time)

    msgan = MSGGAN(LATENT_DIM, beta_1_g, beta_2_d, beta_1_d, beta_2_d, IS_MSG)

    size_list = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)] if IS_MSG else [(64, 64)]

    '''
    msgan.load_model(saved_model_path + "MSGGAN_Generator_" + load_model_id, msgan.generator, max_checkpoints_number)
    msgan.load_model(saved_model_path + "MSGGAN_Discriminator_" + load_model_id, msgan.discriminator, max_checkpoints_number)
    '''
    ds_len = len(ds)
    for epoch in range(EPOCHS):
        start = time.time()

        random.shuffle(ds)
        tf.print("Epoch: ", epoch)

        l = epoch / EPOCHS
        lr_g = lr_g_max - l * (lr_g_max - lr_g_min)
        lr_d = lr_d_max - l * (lr_d_max - lr_d_min)
        # Train
        for n in range(0, ds_len, BATCH_SIZE):
            target_image_b = []
            for i_n in range(n, n + BATCH_SIZE):
                tf.print('.', end='')
                i_n %= ds_len
                if (i_n + 1) % 100 == 0:
                    name_id = "_e" + str(epoch) + "_s" + str(i_n + 1)
                    generate_and_save_samples_l(msgan, save_path="./Projects/GAN_Exp/generated_samples/", name=name_id)
                    tf.print("\n")

                target_image = ds[i_n]
                #target_image = data_augmentation(target_image)
                target_image_b.append(target_image)

            target_image_b = resize_batch_images(target_image_b, size_list=size_list)

            train_dict = msgan.train_step(target_image_b, BATCH_SIZE, lr_g, lr_d)
            
            with summary_writer.as_default():
                for scaler_name in train_dict:
                    tf.summary.scalar(scaler_name, train_dict[scaler_name], step=epoch)
                    tf.print(scaler_name + ": {}".format(train_dict[scaler_name]), end=', ')
                tf.print("\n")

            #generate_and_save_samples_l(msgan, save_path="./Projects/GAN_Exp/generated_samples/")

        # Save the model every certain number epochs
        if (epoch + 1) % save_every_number_episodes == 0:
            save_model(saved_model_path + "MSGGAN_Generator_" + current_time, msgan.generator, max_checkpoints_number)
            save_model(saved_model_path + "MSGGAN_Discriminator_" + current_time, msgan.discriminator, max_checkpoints_number)

        tf.print ('\nTime taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))

# call this function when using GPU with small memory
def using_gpu_memory_growth():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
    return config

if __name__ == "__main__":
    using_gpu_memory_growth()
    train()

