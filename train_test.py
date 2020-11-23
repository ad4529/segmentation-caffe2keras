import argparse
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model
from get_data_paths import get_data_paths
from data_gen import DataGenerator
from create_model import create_model
from get_data_paths import get_data_paths
from custom_loss_and_metrics import soft_dice_loss, mean_iou


def do_train(batch_size, save_name):
    # Load the Train Img-GT Paths
    img2gt_paths = get_data_paths(train=True)

    # Initialize Data Generator
    generator = DataGenerator(img2gt_paths, batch_size=batch_size)

    # Create Model and Transfer Caffe weights
    model = create_model()

    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss=soft_dice_loss, metrics=['accuracy', mean_iou])
    model.fit_generator(generator=generator,
                        steps_per_epoch=int(np.floor(len(img2gt_paths)/batch_size)),
                        epochs=8, use_multiprocessing=True, workers=8)
    model.save(save_name)


def do_eval(model_path):
    # Load saved Model
    model = load_model(model_path,
                       custom_objects={'soft_dice_loss': soft_dice_loss, 'mean_iou': mean_iou})
    # Load Test Img-GT Paths
    img2gt_paths = get_data_paths(train=False)

    # Initialize DataGenerator
    generator = DataGenerator(img2gt_paths, batch_size=1)

    loss, acc, m_iou = model.evaluate_generator(generator, steps=len(img2gt_paths))

    print('Pixel level accuracy: {}'.format(acc))
    print('Mean IoU: {}'.format(m_iou))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Test Model')
    parser.add_argument('--mode', type=str, help='train or test')
    parser.add_argument('--model_path', type=str, default='', help='Saved model location for testing')
    parser.add_argument('--save_name', type=str, default='test_model.h5', help='Save model file')
    parser.add_argument('--cpu_only', type=bool, default=False, help='Specify False to train on a GPU')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')

    args = parser.parse_args()

    if args.mode == 'train':
        if args.cpu_only:
            with tf.device('cpu:0'):
                do_train(args.batch_size, args.save_name)
        else:
            do_train(args.batch_size, args.save_name)
    elif args.mode == 'test':
        if args.model_path == '':
            raise Exception('Specify saved model path to load when evaluating')
        do_eval(args.model_path)
    else:
        raise Exception('Invalid mode {} entered. '
                        'Use mode \'train\' for training and \'test\' for testing'.format(args.mode))