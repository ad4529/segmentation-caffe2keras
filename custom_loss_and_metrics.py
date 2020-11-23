from keras import backend as K


def soft_dice_loss(y_true, y_pred, smooth=1):
    '''
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''

    # skip the batch axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)

    return 1 - K.mean((numerator + smooth) / (denominator + smooth))  # average over classes and batch


def mean_iou(y_true, y_pred, smooth=1):
    axes = tuple(range(1, len(y_pred.shape)))
    intersection = K.sum(K.abs(y_true*y_pred), axes)
    union = K.sum(y_true, axes) + K.sum(y_pred, axes) - intersection
    iou = K.mean((intersection+smooth)/(union+smooth), axis=0)
    return iou
