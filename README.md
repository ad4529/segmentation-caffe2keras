# Transfer Learning SegNet from Caffe to Keras (Segmentation) - SkySpecs Code Sample

The premise of the project is to show that pre-trained weights taken from a network trained on a similar dataset can really help your model to converge faster and better than training from scratch using uninitialized weights or generic weights like ImageNet or MSCOCO. It is also demonstrated that neural networks are essentially platform agnostic and saved weights from any of the DL API's can be used in another. It can however be troublesome and may involve a lot of man-hours!  

SegNet [Paper](https://arxiv.org/pdf/1511.00561.pdf) is a Semantic Segmentation model. My goal was to use this model to train on the [Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) dataset. Using the author provided pre-trained weights of basic segnet (a mini-SegNet) on their Github [here](https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Example_Models/segnet_model_zoo.md) of the Camvid Dataset, it is used in a Keras implementation by keeping all the layers same.  

| ![Camvid Instance](img1.png?raw=True "Instance1") |
|:--:|
| *Camvid dataset instance* |

| ![Cityscapes Instance](img2.png?raw=True "Instance2") |
|:--:|
| *Cityscapes dataset instance* |

It should be pointed out that the weights obtained were of Basic Segnet (with 4 encoder-decoder stages instead of 5 and half the convolution layers in each stage) and not the actual Segnet. It is shown that by using Camvid weights as weight initializations, a basic SegNet outperforms a SegNet (trained from scratch) on the Cityscapes dataset with only half the trainable parameters. Training time Mean-IoU was found to be 63.6% compared to SegNet's 53.6%.  

The model overfit on the training set which is why testing accuracy was low. But that is a problem that can be solved with some paramter tuning, early-stopping, augmentation etc. etc. which is a different endeavor.

## Requirements

* BVLC Caffe built from source (**Not required** per-se as script reads the caffe saved weights file if present which is provided). Check the [repo](https://github.com/BVLC/caffe) here for installation instructions. Beware of a 1000 different dependencies when building!

* TF<=1.14 (Used 1.14 for this project)
* Keras<=2.0.8 (Used 2.0.8 for this project)

### Pip packages

* Numpy
* Other dependencies which may be required by TF and Keras depending on your system.

## Cityscapes Dataset download

The dataset can be found [here](https://www.cityscapes-dataset.com/downloads/). Download and extract the Ground-Truth files `gtFine_trainvaltest.zip` and RGB images `leftImg8bit_trainvaltest.zip` to the root of this directory.

## Usage

### Training

	python3 train_test.py --mode train --save_name <new model save path> --cpu_only <True or False> --batch_size <batch-size for training>

### Testing
	python3 train_test.py --mode test --model_path <saved model path to load>

## Troubleshooting

* If you get an error like `AttributeError: 'str' object has no attribute 'decode'` while testing, you might be using `h5py3.0.0`. You may need to downgrade to `2.X.X` or upgrade to `3.1.X`. It can be done by `pip3 install h5py<3.0.0`.

## Loss and Evaluation Metrics

A custom implementation of F-1 dice score was used as a loss function. The metrics for evaluating were pixel level accuracy and Mean IoU scores.

## Note

The entire idea was conceptualized and coded from scratch in 2 days. There may be some things that are not quite efficient or could have been done better if some more time was spent on this. Some things that come to mind are:

* Use a `.YAML` config file to store all the hyper-parameters and dataset/weights path for easy usage.
* Hyper-parameter search to prevent overfitting
* Using model checkpoints to resume training.
* Having a pretty demo for test images.




