import keras
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.models import Model
import numpy as np
import os


def get_weights():
    # Get the saved weights from Caffe
    # If already stored, use them
    weights_file = 'Camvid_weights.npy'
    if not os.path.isfile(weights_file):
        from get_caffe_weights import get_caffe_weights
        get_caffe_weights()

    weights = np.load(weights_file, allow_pickle=True, encoding='bytes')

    # Numpy saves strings as bytes. Need to convert byte keys(layer names) to utf-8
    weights_act = {}
    for layer in weights[()]:
        weights_act[layer.decode('utf-8')] = weights[()][layer]

    print('Loaded caffe weights')
    return weights_act


def create_model():
    # Model created by adapting the layer information available in the .prototxt caffe model file

    print('Creating model skeleton')
    # Input
    inputs = layers.Input(shape=(360,480,3), name='Input')

    # Encoder Stage 1
    conv1 = layers.Conv2D(64, (7,7), padding='same', name='conv1')(inputs)
    conv1_bn = BatchNormalization(name='conv1_bn')(conv1)
    relu_1 = Activation('relu', name='Relu_1')(conv1_bn)
    pool_1 = layers.MaxPooling2D((2,2), strides=2)(relu_1)

    # Encoder Stage 2
    conv2 = layers.Conv2D(64, (7, 7), padding='same', name='conv2')(pool_1)
    conv2_bn = BatchNormalization(name='conv2_bn')(conv2)
    relu_2 = Activation('relu', name='Relu_2')(conv2_bn)
    pool_2 = layers.MaxPooling2D((2, 2), strides=2)(relu_2)

    # Encoder Stage 3
    conv3 = layers.Conv2D(64, (7, 7), padding='same', name='conv3')(pool_2)
    conv3_bn = BatchNormalization(name='conv3_bn')(conv3)
    relu_3 = Activation('relu')(conv3_bn)
    pool_3 = layers.MaxPooling2D((2, 2), strides=2)(relu_3)

    # Encoder Stage 4
    conv4 = layers.Conv2D(64, (7, 7), padding='same', name='conv4')(pool_3)
    conv4_bn = BatchNormalization(name='conv4_bn')(conv4)
    relu_4 = Activation('relu')(conv4_bn)
    pool_4 = layers.MaxPooling2D((2, 2), strides=2)(relu_4)

    # Decoder Stage 4
    upsample_4 = layers.UpSampling2D()(pool_4)
    conv4_decode = layers.Conv2D(64, (7, 7), padding='same', name='conv_decode4')(upsample_4)
    conv4dec_bn = BatchNormalization(name='conv_decode4_bn')(conv4_decode)

    # Decoder Stage 3
    upsample_3 = layers.UpSampling2D()(conv4dec_bn)
    conv3_decode = layers.Conv2D(64, (7, 7), padding='same', name='conv_decode3')(upsample_3)
    conv3dec_bn = layers.BatchNormalization(name='conv_decode3_bn')(conv3_decode)

    # Decoder Stage 2
    upsample_2 = layers.UpSampling2D()(conv3dec_bn)
    conv2_decode = layers.Conv2D(64, (7, 7), padding='same', name='conv_decode2')(upsample_2)
    conv2dec_bn = layers.BatchNormalization(name='conv_decode2_bn')(conv2_decode)

    # Decoder Stage 1
    upsample_1 = layers.UpSampling2D()(conv2dec_bn)
    conv1_decode = layers.Conv2D(64, (7, 7), padding='same', name='conv_decode1')(upsample_1)
    conv1dec_bn = layers.BatchNormalization(name='conv_decode1_bn')(conv1_decode)

    # Logits Layer
    logits = layers.Conv2D(30, (1, 1), padding='same', activation='softmax', name='Logits')(conv1dec_bn)

    model = Model(inputs=inputs, outputs=logits)
    # model.summary()

    # Transfer weights from caffe
    weights = get_weights()

    for layer in model.layers:
        if 'conv' in layer.name:
            weight = np.asarray(weights[layer.name][0])
            # Change from Caffe's NumFil x C x H X W to Keras' H x W x C x NumFil
            weight = np.squeeze(np.transpose(weight, (2,3,1,0)))

            # No need to change bias
            bias = np.squeeze(np.asarray(weights[layer.name][1]))

            # Batch Norm weights in Keras require beta and gamma along with weight
            # and bias whereas caffe does not store beta gamma (weird). Thus we
            # keep Beta and Gamma with base initializations.
            if 'bn' in layer.name:
                beta = layer.get_weights()[2]
                gamma = layer.get_weights()[3]
                layer.set_weights([weight, bias, beta, gamma])
            else:
                layer.set_weights([weight, bias])

    print("Done Setting Weights")
    model.summary()
    return model


if __name__ == '__main__':
    create_model()
