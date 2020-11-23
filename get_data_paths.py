import os
import glob
import random


def get_data_paths(train=True):
    if train:
        imgs_paths = glob.glob('leftImg8bit_trainvaltest/leftImg8bit/train/*/*.png')
        imgs_paths += glob.glob('leftImg8bit_trainvaltest/leftImg8bit/val/*/*.png')
        gt_paths = glob.glob('gtFine_trainvaltest/gtFine/train/*/*labelIds.png')
        gt_paths += glob.glob('gtFine_trainvaltest/gtFine/val/*/*labelIds.png')
    else:
        imgs_paths = glob.glob('leftImg8bit_trainvaltest/leftImg8bit/test/*/*.png')
        gt_paths = glob.glob('gtFine_trainvaltest/gtFine/test/*/*labelIds.png')

    tot_imgs = len(imgs_paths)

    # Create a dictionary of Ids and Image Paths
    img_ids = [os.path.basename(p).split('_') for p in imgs_paths]
    img_ids = [p[0]+'_'+p[1]+'_'+p[2] for p in img_ids]

    id2img_paths = {img_ids[idx]: imgs_paths[idx] for idx in range(tot_imgs)}

    # Create a dictionary of Ids and GT Paths
    gt_ids = [os.path.basename(p).split('_') for p in gt_paths]
    gt_ids = [p[0] + '_' + p[1] + '_' + p[2] for p in gt_ids]

    id2gt_paths = {gt_ids[idx]: gt_paths[idx] for idx in range(tot_imgs)}

    # Now create ImgPath-GTPath pairs
    img2gt_paths = {id2img_paths[idx]: id2gt_paths[idx] for idx in id2img_paths}

    # Shuffle the dictionary for better training
    keys = list(img2gt_paths.keys())
    random.shuffle(keys)
    img2gt_paths = {key: img2gt_paths[key] for key in keys}

    return img2gt_paths


if __name__ == '__main__':
    get_data_paths(train=False)
