import caffe
import numpy as np


def get_caffe_weights(proto='segnet_basic_camvid.prototxt', model='segnet_basic_camvid.caffemodel'):
    net = caffe.Net(proto, model, caffe.TEST) # Test mode gives us the final trained weights from Camvid
    weights = {}
    for layer in net._layer_names:
        if 'conv' in layer:
            weight = net.params[layer][0].data[...]
            bias = net.params[layer][1].data[...]
            weights[layer] = [weight, bias]

    np.save('Camvid_weights.npy', weights)


if __name__ == '__main__':
    get_caffe_weights()
