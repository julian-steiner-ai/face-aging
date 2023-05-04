"""
Module implements a Conditiong GAN for Face Aging.

Paper: Face Aging With Identity-Preserved Conditional Generative Adversarial Networks
"""

import pickle as pkl

import numpy as np

from os.path import join

from keras import Input, Model
from keras.backend import expand_dims, tile, l2_normalize, resize_images
from keras.applications import InceptionResNetV2
from keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from keras.layers import Reshape, concatenate, LeakyReLU, Lambda
from keras.layers import Activation, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

from loguru import logger

def expand_label_input(x):
    x = expand_dims(x, axis = 1)
    x = expand_dims(x, axis = 1)
    x = tile(x, [1, 32, 32, 1])
    return x

class IPCGAN:
    """
    CGAN implementation for Face Aging.
    """
    @staticmethod
    def load(directory):
        """
        Load Generative Adversarial Network from Pickle-File.
        """
        with open(join(directory, 'params.pkl'), 'rb') as file:
            params = pkl.load(file)
        gan = IPCGAN(*params)
        gan.load_weights(
            join(directory, 'weights/discriminator/weights.h5'),
            join(directory, 'weights/generator/weights.h5')
        )
        return gan
    
    def __init__(self):
        self._init_encoder()
        self._init_discriminator()
        self._init_generator()
        self._init_network()

    def _init_encoder(self):
        input_layer = Input(shape = (64, 64, 3))

        ## 1st Convolutional Block
        enc = Conv2D(filters = 32, kernel_size = 5, strides = 2, padding = 'same')(input_layer)
        enc = LeakyReLU(alpha = 0.2)(enc)

        ## 2nd Convolutional Block
        enc = Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same')(enc)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(alpha = 0.2)(enc)

        ## 3rd Convolutional Block
        enc = Conv2D(filters = 128, kernel_size = 5, strides = 2, padding = 'same')(enc)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(alpha = 0.2)(enc)

        ## 4th Convolutional Block
        enc = Conv2D(filters = 256, kernel_size = 5, strides = 2, padding = 'same')(enc)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(alpha = 0.2)(enc)

        ## Flatten layer
        enc = Flatten()(enc)
        
        ## 1st Fully Connected Layer
        enc = Dense(4096)(enc)
        enc = BatchNormalization()(enc)
        enc = LeakyReLU(alpha = 0.2)(enc)

        ## 2nd Fully Connected Layer
        enc = Dense(100)(enc)

        ## Create a model
        self.encoder = Model(inputs = [input_layer], outputs = [enc])

    def _init_generator(self):
        latent_dims = 100
        num_classes = 6

        input_z_noise = Input(shape = (latent_dims, ))
        input_label = Input(shape = (num_classes, ))

        x = concatenate([input_z_noise, input_label])

        x = Dense(2048, input_dim = latent_dims + num_classes)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dropout(0.2)(x)

        x = Dense(256 * 8 * 8)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dropout(0.2)(x)

        x = Reshape((8, 8, 256))(x)

        x = UpSampling2D(size = (2, 2))(x)
        x = Conv2D(filters = 128, kernel_size = 5, padding = 'same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = UpSampling2D(size = (2, 2))(x)
        x = Conv2D(filters = 64, kernel_size = 5, padding = 'same')(x)
        x = BatchNormalization(momentum = 0.8)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        x = UpSampling2D(size = (2, 2))(x)
        x = Conv2D(filters = 3, kernel_size = 5, padding = 'same')(x)
        x = Activation('tanh')(x)

        self.generator = Model(inputs = [input_z_noise, input_label], outputs = [x])
    
    def _init_discriminator(self):
        input_shape = (64, 64, 3)
        label_shape = (6, )
        image_input = Input(shape = input_shape)
        label_input = Input(shape = label_shape)
        
        x = Conv2D(64, kernel_size = 3, strides = 2, padding = 'same')(image_input)
        x = LeakyReLU(alpha = 0.2)(x)
        
        label_input1 = Lambda(
            expand_label_input
        )(label_input)

        x = concatenate([x, label_input1], axis = 3)
        
        x = Conv2D(128, kernel_size = 3, strides = 2, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Conv2D(256, kernel_size = 3, strides = 2, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Conv2D(512, kernel_size = 3, strides = 2, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha = 0.2)(x)
        
        x = Flatten()(x)
        x = Dense(1, activation = 'sigmoid')(x)
        
        self.discriminator = Model(inputs = [image_input, label_input], outputs = [x])
    
    def _build_fr_combined_network(self, encoder, generator, fr_model):
        """
        Face Recognition Combined Network
        """
        input_image = Input(shape = (64, 64, 3))
        input_label = Input(shape = (6, ))
        
        latent0 = encoder(input_image)
        
        gen_images = generator([latent0, input_label])
        
        fr_model.trainable = False
        
        resized_images = Lambda(lambda x: resize_images(
            gen_images,
            height_factor = 2,
            width_factor = 2,
            data_format = 'channels_last')
        )(gen_images)
        
        embeddings = fr_model(resized_images)
        
        self.face_recognition_combined = Model(inputs = [input_image, input_label], outputs = [embeddings])
    
    def _build_fr_model(self, input_shape):
        """
        Face Recogntion Network.
        """
        resnet_model = InceptionResNetV2(
            include_top = False,
            weights = 'imagenet',
            input_shape = input_shape,
            pooling = 'avg'
        )

        image_input = resnet_model.input
        x = resnet_model.layers[-1].output
        out = Dense(128)(x)
        embedder_model = Model(inputs = [image_input], outputs = [out])

        input_layer = Input(shape = input_shape)

        x = embedder_model(input_layer)
        output = Lambda(lambda x: l2_normalize(x, axis = -1))(x)

        self.face_recognition = Model(inputs = [input_layer], outputs = [output])

    def _build_image_resizer(self):
        input_layer = Input(shape = (64, 64, 3))
  
        resized_images = Lambda(lambda x: resize_images(
            x,
            height_factor = 3,
            width_factor = 3,
            data_format = 'channels_last')
        )(input_layer)
        
        self.image_resizer = Model(inputs = [input_layer], outputs = [resized_images])
    
    def _init_network(self):
        self.discriminator.compile(
            loss=['binary_crossentropy'],
            optimizer=Adam(
                lr = 0.0002,
                beta_1 = 0.5,
                beta_2 = 0.999,
                epsilon = 10e-8
            )
        )

        self.generator.compile(
            loss=['binary_crossentropy'],
            optimizer=Adam(
                lr = 0.0002,
                beta_1 = 0.5,
                beta_2 = 0.999,
                epsilon = 10e-8
            )
        )

        self.discriminator.trainable = False

        input_z_noise = Input(shape = (100, ))
        input_label = Input(shape = (6, ))

        recons_images = self.generator([input_z_noise, input_label])
        valid = self.discriminator([recons_images, input_label])

        self.adversarial_model = Model(
            inputs = [input_z_noise, input_label],
            outputs = [valid]
        )

        self.adversarial_model.compile(
            loss = ['binary_crossentropy'],
            optimizer= Adam(
            lr = 0.0002,
            beta_1 = 0.5,
            beta_2 = 0.999,
            epsilon = 10e-8)
        )

    def train(
            self,
            epochs,
            x_train,
            y_train,
            batch_size,
            z_shape,
            real_labels,
            fake_labels,
            train_gan=True,
            train_encoder=False,
            train_gan_with_fr=False):
        
        if train_gan:
            self._train_gan(
                epochs=epochs,
                x_train=x_train,
                y_train=y_train,
                batch_size=batch_size,
                z_shape=z_shape,
                real_labels=real_labels,
                fake_labels=fake_labels
            )
    
    def _train_gan(
            self,
            epochs,
            x_train,
            y_train,
            batch_size,
            z_shape,
            real_labels,
            fake_labels):
        for epoch in range(epochs):
            logger.info(f"Epoch: {epoch}")
            
            gen_losses = []
            dis_losses = []
            
            number_of_batches = int(len(x_train) / batch_size)
            logger.info(f"Number of batches: {number_of_batches}")

            for index in range(number_of_batches):
                logger.info(f'Batch: {index + 1}')
                
                images_batch = x_train[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)
                
                y_batch = y_train[index * batch_size: (index + 1) * batch_size]
                z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
                
                initial_recons_images = self.generator.predict_on_batch([z_noise, y_batch])
                
                d_loss_real = self.discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = self.discriminator.train_on_batch([initial_recons_images, y_batch], fake_labels)
                
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                logger.info(f'd_loss: {d_loss}')
                
                z_noise2 = np.random.normal(0, 1, size = (batch_size, z_shape))
                random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                random_labels = to_categorical(random_labels, 6)
                
                g_loss = self.adversarial_model.train_on_batch([z_noise2, random_labels],  np.asanyarray([1] * batch_size))
                
                logger.info(f'g_loss: {g_loss}')
                
                gen_losses.append(g_loss)
                dis_losses.append(d_loss)
                
            logger.info(f'g_loss: {np.mean(gen_losses)} - Epoch: {epoch}')
            logger.info(f'd_loss: {np.mean(dis_losses)} - Epoch: {epoch}')
            
            if epoch % 10 == 0:
                images_batch = x_train[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)
                
                y_batch = y_train[0:batch_size]
                z_noise = np.random.normal(0, 1, size = (batch_size, z_shape))
                
                try:
                    self.discriminator.save_weights(join('./model/face_aging', 'weights', 'discriminator', 'weights-%d.h5' % (epoch)))
                    self.discriminator.save_weights(join('./model/face_aging', 'weights', 'discriminator', 'weights.h5'))
                    self.generator.save_weights(join('./model/face_aging', 'weights', 'generator', 'weights-%d.h5' % (epoch)))
                    self.generator.save_weights(join('./model/face_aging', 'weights', 'generator', 'weights.h5'))
                    self.save()
                except Exception as e:
                    logger.info(f'Error: {e}')

    def generate(self, n_images, ages, z_shape):
        y_batch = np.asanyarray(ages)
        z_noise = np.random.normal(0, 1, size = (n_images, z_shape))
        
        return self.generator.predict_on_batch([z_noise, y_batch])
    
    def save(self):
        """
        Save the FaceAging-GAN for loading it again. 
        """
        self.discriminator.save(join('./model/face_aging', 'discriminator.h5'))
        self.generator.save(join('./model/face_aging', 'generator.h5'))
        pkl.dump([], open(join('./model/face_aging', 'params.pkl'), 'wb'))
        pkl.dump(self, open(join('./model/face_aging', "obj.pkl"), "wb"))

    def load_weights(self, discriminator_filepath, generator_filepath):
        """
        Load the saved weights.
        """
        self.discriminator.load_weights(discriminator_filepath)
        self.generator.load_weights(generator_filepath)