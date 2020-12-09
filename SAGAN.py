'''
    Self-Attention GAN on cifar10 dataset
    CIFAR dataset: https://www.cs.toronto.edu/~kriz/cifar.html
'''
import os
import time
import datetime

import tensorflow as tf
from tensorflow.keras import layers, models, initializers, constraints, optimizers
from tensorflow_addons.layers import InstanceNormalization
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def l2_normalize(x, eps=1e-12):
    '''
    Scale input by the inverse of it's euclidean norm
    '''
    return x / tf.linalg.norm(x + eps)

class Spectral_Norm(constraints.Constraint):
    '''
    Uses power iteration method to calculate a fast approximation 
    of the spectral norm (Golub & Van der Vorst)
    The weights are then scaled by the inverse of the spectral norm
    '''
    def __init__(self, power_iters=5):
        self.n_iters = power_iters

    def __call__(self, w):
        flattened_w = tf.reshape(w, [w.shape[0], -1])
        u = tf.random.normal([flattened_w.shape[0]])
        v = tf.random.normal([flattened_w.shape[1]])
        for i in range(self.n_iters):
            v = tf.linalg.matvec(tf.transpose(flattened_w), u)
            v = l2_normalize(v)
            u = tf.linalg.matvec(flattened_w, v)
            u = l2_normalize(u)
        sigma = tf.tensordot(u, tf.linalg.matvec(flattened_w, v), axes=1)
        return w / sigma

    def get_config(self):
        return {'n_iters': self.n_iters}

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Conv2D(128, kernel_size=(4,4), strides=(2,2), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Conv2D(256, kernel_size=(4,4), strides=(2,2), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_constraint=Spectral_Norm()))
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, kernel_constraint=Spectral_Norm()))
    return model

class ResnetBlockGen(tf.keras.Model):
    def __init__(self, kernel_size, filters, pad='same'):
        super(ResnetBlockGen, self).__init__(name='')
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.deconv2a = tf.keras.layers.Conv2DTranspose(filters, kernel_size,
                                                        padding=pad)
        
        
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.deconv2b = tf.keras.layers.Conv2DTranspose(filters, kernel_size,
                                                        padding=pad)
        
        self.up_sample = tf.keras.layers.UpSampling2D(size=(2,2))
        self.shortcut_conv = tf.keras.layers.Conv2DTranspose(filters, 
                                                            kernel_size=1,
                                                            padding=pad)
        
    def call(self, input_tensor, training=False):
        x = self.bn1(input_tensor)
        x = tf.nn.relu(x)
        x = self.up_sample(x)
        x = self.deconv2a(x)
        
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.deconv2b(x)
        
        sc_x = self.up_sample(self.shortcut_conv(input_tensor))
        return x + sc_x

class ResnetBlockDisc(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, downsample=False, pad='same'):
        super(ResnetBlockDisc, self).__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding=pad,
                                            kernel_initializer='glorot_uniform',
                                            kernel_constraint=Spectral_Norm())
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding=pad,
                                            kernel_initializer='glorot_uniform',
                                            kernel_constraint=Spectral_Norm())
        self.shortcut_conv = tf.keras.layers.Conv2D(filters, kernel_size=(1,1),
                                                    kernel_initializer='glorot_uniform',
                                                    padding=pad,
                                                    kernel_constraint=Spectral_Norm())
        self.downsample_layer = tf.keras.layers.AvgPool2D((2,2))
        self.downsample = downsample

    def residual(self, x):
        h = x
        h = tf.nn.relu(h)
        h = self.conv1(h)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        if self.downsample:
            h = self.downsample_layer(h)
        return h
    
    def shortcut(self, x):
        h2 = x
        if self.downsample:
            h2 = self.downsample_layer(x)
        return self.shortcut_conv(h2)
    
    def call(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size=3, pad='same'):
        super(OptimizedBlock, self).__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding=pad,
                                            kernel_initializer='glorot_uniform',
                                            kernel_constraint=Spectral_Norm())
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding=pad,
                                            kernel_initializer='glorot_uniform',
                                            kernel_constraint=Spectral_Norm())
        self.shortcut_conv = tf.keras.layers.Conv2D(filters, kernel_size=(1,1),
                                                    kernel_initializer='glorot_uniform',
                                                    padding=pad,
                                                    kernel_constraint=Spectral_Norm())
        self.downsample_layer = tf.keras.layers.AvgPool2D((2,2))
        
    def residual(self, x):
        h = x
        h = self.conv1(h)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        h = self.downsample_layer(h)
        return h
    
    def shortcut(self, x):
        return self.shortcut_conv(self.downsample_layer(x))
    
    def call(self, x):
        return self.residual(x) + self.shortcut(x)

class SelfAttentionBlock(tf.keras.Model):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()
        self.sigma = K.variable(0.0, name='sigma')
        self.phi_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.g_pool   = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        
    def build(self, inp):
        batch_size, h, w, n_channels = inp
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.h = h
        self.w = w
        
        self.location_num = h*w
        self.downsampled_num = self.location_num // 4
        
        self.theta = tf.keras.layers.Conv2D(filters = n_channels // 8, 
                                            kernel_size=[1, 1], 
                                            padding= 'same',
                                            strides=(1,1))
        self.phi   = tf.keras.layers.Conv2D(filters = n_channels // 8,
                                            kernel_size=[1, 1],
                                            padding='same',
                                            strides=(1,1))
        self.attn_conv = tf.keras.layers.Conv2D(self.n_channels, kernel_size=[1, 1])
        
        self.g = tf.keras.layers.Conv2D(filters = n_channels // 2, 
                                                kernel_size=[1, 1])
        
    def call(self, x):
        theta = self.theta(x)
        theta = tf.reshape(theta, [self.batch_size, self.location_num,
                                self.n_channels // 8])
        
        phi = self.phi(x)
        phi = self.phi_pool(phi)
        phi = tf.reshape(phi, [self.batch_size, self.downsampled_num,
                            self.n_channels // 8])

        
        attn = tf.matmul(theta, phi, transpose_b=True)
        attn = tf.nn.softmax(attn)
        
        # g path
        g = self.g(x)
        g = self.g_pool(g)
        g = tf.reshape(
        g, [self.batch_size, self.downsampled_num, self.n_channels // 2])
        
        attn_g = tf.matmul(attn, g)
        attn_g = tf.reshape(attn_g, [self.batch_size, self.h, self.w,
                                    self.n_channels // 2])
        attn_g = self.attn_conv(attn_g)
        return x + (attn_g * self.sigma)

def make_resnet_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16*256, kernel_initializer='glorot_uniform'))
    model.add(tf.keras.layers.Reshape((4, 4, 256)))
    model.add(ResnetBlockGen(3, 256))
    model.add(ResnetBlockGen(3, 256, pad='same')) 
    model.add(SelfAttentionBlock())
    model.add(ResnetBlockGen(3, 256, pad='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=3, strides=(1,1),
                                        padding='same', activation='tanh'))
    return model

def make_resnet_discriminator_model():
    model = tf.keras.Sequential()
    model.add(OptimizedBlock(128))
    model.add(ResnetBlockDisc(128, downsample=True))
    model.add(ResnetBlockDisc(128, downsample=False))
    model.add(ResnetBlockDisc(128, downsample=False))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.GlobalAvgPool2D(data_format='channels_last' ))
    model.add(tf.keras.layers.Dense(1, kernel_initializer='glorot_uniform',
                                    kernel_constraint=Spectral_Norm()))
    return model

class ConditionalBatchNorm(layers.Layer):
    def __init__(self, num_categories, decay_rate=0.999, 
                center=True, scale=True):
        super(ConditionalBatchNorm, self).__init__()
        self.num_categories = num_categories
        self.center = center
        self.scale = scale
        self.decay_rate = decay_rate
        
    def build(self, input_size):
        self.inputs_shape = tf.TensorShape(input_size)
        params_shape = self.inputs_shape[-1:]
        axis = [0, 1, 2]
        shape = tf.TensorShape([self.num_categories]).concatenate(params_shape)
        moving_shape = tf.TensorShape([1,1,1]).concatenate(params_shape)
        
        self.gamma = self.add_variable(name='gamma', shape=shape,
                                    initializer='ones')
        self.beta  = self.add_variable(name='beta', shape=shape,
                                    initializer='zeros')
        
        self.moving_mean = self.add_variable(name='mean',
                                            shape=moving_shape,
                                            initializer='zeros',
                                            trainable=False)
        self.moving_var  = self.add_variable(name='var',
                                            shape=moving_shape,
                                            initializer='ones', 
                                            trainable=False)
        
        
    def call(self, inputs, labels, is_training=True):
        beta = tf.gather(self.beta, labels)
        beta = tf.expand_dims(tf.expand_dims(beta, 1), 1)
        gamma = tf.gather(self.gamma, labels)
        gamma = tf.expand_dims(tf.expand_dims(gamma, 1), 1)
        decay = self.decay_rate
        variance_epsilon = 1e-5
        axis = [0, 1, 2]
        if is_training:
            mean, variance = tf.nn.moments(inputs, axis, keepdims=True)
            self.moving_mean.assign(self.moving_mean * decay + mean * (1 - decay))
            self.moving_var.assign(self.moving_var * decay + variance * (1 - decay))
            outputs = tf.nn.batch_normalization(
                inputs, mean, variance, beta, gamma, variance_epsilon)
        else:
            outputs = tf.nn.batch_normalization(
                inputs, self.moving_mean, self.moving_var, 
                beta, gamma, variance_epsilon)
        outputs.set_shape(self.inputs_shape)
        return outputs

class Generator_CBN(tf.keras.Model):
    def __init__(self, n_classes, training=True):
        super(Generator_CBN, self).__init__()
        self.linear1 = layers.Dense(4*4*512, use_bias=False,
                                    kernel_initializer='glorot_uniform')
        self.reshape = layers.Reshape([4, 4, 512])
        
        self.cbn1 = ConditionalBatchNorm(n_classes)
        self.deconv1 = layers.Conv2DTranspose(filters=256, kernel_size=(4,4),
                                            strides=2, padding='same')
        
        self.cbn2 = ConditionalBatchNorm(n_classes)
        self.deconv2 = layers.Conv2DTranspose(filters=128, kernel_size=(4,4),
                                            strides=2, padding='same')
        
        self.cbn3 = ConditionalBatchNorm(n_classes)
        self.deconv3 = layers.Conv2DTranspose(filters=64, kernel_size=(4,4),
                                            strides=2, padding='same')
        
        self.cbn4 = ConditionalBatchNorm(n_classes)
        self.deconv4 = layers.Conv2DTranspose(filters=3, kernel_size=(3,3),
                                            strides=1, use_bias=True,
                                            padding='same')

    
    def call(self, inp, labels):
        x = self.linear1(inp)
        x = self.reshape(x)
        
        x = self.cbn1(x, labels)
        x = tf.nn.relu(x)
        x = self.deconv1(x)
        
        x = self.cbn2(x, labels)
        x = tf.nn.relu(x)
        x = self.deconv2(x)
        
        x = self.cbn3(x, labels)
        x = tf.nn.relu(x)
        x = self.deconv3(x)
        
        x = self.cbn4(x, labels)
        x = tf.nn.relu(x)
        x = self.deconv4(x)
        
        x = tf.nn.tanh(x)
        return x

def make_cbn_generator_model():
    model = Generator_CBN(n_classes=10)
    return model

class SelfModulationBN(layers.Layer):
    def __init__(self, z_size=128, decay_rate=0.999, 
                center=True, scale=True):
        super(SelfModulationBN, self).__init__()
        self.z_size = z_size
        self.center = center
        self.scale = scale
        self.decay_rate = decay_rate

        
    def build(self, input_size):
        self.inputs_shape = tf.TensorShape(input_size)
        params_shape = self.inputs_shape[-1:]
        batch_shape = self.inputs_shape[0]
        z_shape = tf.TensorShape(self.z_size)
        mlp_shape = z_shape.concatenate(params_shape)
        moving_shape = tf.TensorShape([batch_shape,1,1]).concatenate(params_shape)
        self.moving_mean = self.add_variable(name='mean',
                                            shape=moving_shape,
                                            initializer='zeros',
                                            trainable=False)
        self.moving_var  = self.add_variable(name='var',
                                            shape=moving_shape,
                                            initializer='ones', 
                                            trainable=False)
        self.ln_gamma = self.add_variable(name='gamma', shape=mlp_shape,
                                    initializer='glorot_normal')
        self.ln_beta  = self.add_variable(name='beta', shape=mlp_shape,
                                    initializer='glorot_normal')
        
    def call(self, inputs, z, is_training=True):
        z = tf.squeeze(z, axis=[1,2])
        gamma = tf.matmul(z, self.ln_gamma)
        gamma = tf.nn.relu(gamma)
        beta = tf.matmul(z, self.ln_beta)
        beta = tf.nn.relu(beta)
        gamma = tf.expand_dims(tf.expand_dims(gamma, axis=1), axis=1)
        beta = tf.expand_dims(tf.expand_dims(beta, axis=1), axis=1)
        decay = self.decay_rate
        variance_epsilon = 1e-5
        axis = [0, 1, 2]
        mean, variance = tf.nn.moments(inputs, axis, keepdims=True)
        self.moving_mean.assign(self.moving_mean * decay + mean * (1 - decay))
        self.moving_var.assign(self.moving_var * decay + variance * (1 - decay))
        outputs = tf.nn.batch_normalization(
            inputs, mean, variance, beta, gamma, variance_epsilon)
        outputs.set_shape(self.inputs_shape)
        return outputs

class Generator_SBN(tf.keras.Model):
    def __init__(self, z_shape, training=True):
        super(Generator_SBN, self).__init__()
        self.linear1 = layers.Dense(4*4*512, use_bias=False,
                                    kernel_initializer='glorot_uniform')
        self.reshape = layers.Reshape([4, 4, 512])
        
        self.sbn1 = SelfModulationBN(z_shape)
        self.deconv1 = layers.Conv2DTranspose(filters=256, kernel_size=(4,4),
                                            strides=2, padding='same')
        
        self.sbn2 = SelfModulationBN(z_shape)
        self.deconv2 = layers.Conv2DTranspose(filters=128, kernel_size=(4,4),
                                            strides=2, padding='same')
        
        self.sbn3 = SelfModulationBN(z_shape)
        self.deconv3 = layers.Conv2DTranspose(filters=64, kernel_size=(4,4),
                                            strides=2, padding='same')
        
        self.sbn4 = SelfModulationBN(z_shape)
        self.deconv4 = layers.Conv2DTranspose(filters=3, kernel_size=(3,3),
                                            strides=1, use_bias=True,
                                            padding='same')
        
    def call(self, inp):
        x = self.linear1(inp)
        x = self.reshape(x)
        
        x = self.sbn1(x, inp)
        x = tf.nn.relu(x)
        x = self.deconv1(x)
        
        x = self.sbn2(x, inp)
        x = tf.nn.relu(x)
        x = self.deconv2(x)
        
        x = self.sbn3(x, inp)
        x = tf.nn.relu(x)
        x = self.deconv3(x)
        
        x = self.sbn4(x, inp)
        x = tf.nn.relu(x)
        x = self.deconv4(x)
        
        x = tf.nn.tanh(x)
        return x

def make_sbn_generator_model():
    model = Generator_SBN(z_shape=128)
    return model

def discriminator_loss(real_output, fake_output):
    L1 = K.mean(K.softplus(-real_output))
    L2 = K.mean(K.softplus(fake_output))
    loss = L1 + L2
    return loss

def discriminator_hinge_loss(real_output, fake_output):
    loss = K.mean(K.relu(1. - real_output))
    loss += K.mean(K.relu(1. + fake_output))
    return loss

def generator_loss(fake_output):
    return K.mean(K.softplus(-fake_output))

def generator_hinge_loss(fake_output):
    return -1 * K.mean(fake_output)

EPOCHS = 50
BATCH_SIZE = 32
noise_dim = 128
num_examples_to_generate = 8
seed = tf.random.normal([BATCH_SIZE, 1, 1, noise_dim])

def generate_and_save_images(model, epoch, test_input, bn_type):
    if bn_type == "cbn":
        label = tf.convert_to_tensor(np.random.randint(0, 10, 128))
        label = tf.squeeze(label)
        predictions = model(test_input, label)
    elif bn_type == "sbn":
        predictions = model(test_input)
    else:
        predictions = model(test_input, training=False) 

    fig = plt.figure(figsize=(4,4))

    for i in range(8):
        plt.subplot(4, 4, i+1)
        plt.imshow((predictions[i]+1)/2)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

@tf.function
def train_step(images, labels=None, sbn=False, disc_steps=1):
    for _ in range(disc_steps):
        with tf.GradientTape() as disc_tape:
            
            noise = tf.random.normal([BATCH_SIZE, 1, 1, noise_dim])
            if labels is not None:
                generated_images = generator(noise, labels)
            else:
                if sbn:
                    generated_images = generator(noise)
                else:
                    generated_images = generator(noise, training=True) 
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)        
            disc_loss = discriminator_hinge_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
    with tf.GradientTape() as gen_tape:
        noise = tf.random.normal([BATCH_SIZE, 1, 1, noise_dim])
        if labels is not None:
            generated_images = generator(noise, labels)
        else:
            if sbn:
                generated_images = generator(noise)
            else:
                generated_images = generator(noise, training=True) 
        fake_output = discriminator(generated_images, training=False)
        gen_loss = generator_hinge_loss(fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    singular_values = tf.linalg.svd(discriminator.trainable_variables[-2])[0]
    condition_number = tf.reduce_max(singular_values)
    train_stats = {'d_loss': disc_loss, "g_loss": gen_loss, 
                   'd_grads': gradients_of_discriminator, 'g_grads': gradients_of_generator,
                   'cond_number': condition_number}
    return train_stats

def train(dataset, epochs, bn_type=None):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    for epoch in range(epochs):
        start = time.time()
        
        for batch in dataset:
            if bn_type == "cbn": 
                image_batch, label_batch = batch
                label_batch = tf.squeeze(label_batch)
                train_diction = train_step(image_batch, label_batch)
            elif bn_type == "sbn":
                image_batch = batch
                train_diction = train_step(image_batch, sbn=True)
            else:
                image_batch= batch
                train_diction = train_step(image_batch)


        with train_summary_writer.as_default():
            tf.summary.scalar('disc_total_loss', train_diction['d_loss'], step=epoch)
            tf.summary.scalar('gen_total_loss', train_diction['g_loss'], step=epoch)
            tf.summary.scalar("max disc_grads[-1]: ", np.max(train_diction['d_grads'][-1].numpy()), step=epoch)
            tf.summary.scalar("max disc_grads[0]: ", np.max(train_diction['d_grads'][0].numpy()), step=epoch)
            tf.summary.scalar('gen_grads', np.max(train_diction['g_grads'][-1].numpy()), step=epoch)
            tf.summary.scalar('condition_number', train_diction['cond_number'], step=epoch)

        generate_and_save_images(generator,
                                epoch + 1,
                                seed, bn_type)

        # Save the model every 15 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # Generate after the final epoch
    generate_and_save_images(generator,
                            epochs,
                            seed, bn_type)
 
def SAGAN_train():
    global generator, discriminator, generator_optimizer, discriminator_optimizer

    discriminator = make_resnet_discriminator_model()
    generator = make_resnet_generator_model()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        2e-4, decay_rate=0.99, decay_steps=50000*EPOCHS)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0, beta_2=0.9)
    
    train_dataset = load_cifar10_dataset()
    train(train_dataset, EPOCHS)
    # train(train_dataset, EPOCHS, bn_type='cbn'); make_cbn_generator_model
    # train(train_dataset, EPOCHS, bn_type='sbn'); make_sbn_generator_model
    # make_resnet_discriminator_model; loss to hinge 

def load_cifar10_dataset(sbn=False):
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images.reshape(train_images.shape[0], 32, 32, 3).astype('float32')
    train_images = (train_images/255) * 2 - 1
    train_labels = train_labels.astype('int32')

    BUFFER_SIZE = 50000
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    if sbn:
        image_dataset = tf.data.Dataset.from_tensor_slices(train_images)
        label_dataset = tf.data.Dataset.from_tensor_slices(train_labels)
        train_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
        train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
        train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset

if __name__ == "__main__":
    SAGAN_train()