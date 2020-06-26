import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb


def main():
    img_size = 256

    paths = glob('data/2018 Data Science Bowl_stage1_train/*')

    os.makedirs('data/dsb2018/images', exist_ok=True)
    os.makedirs('data/dsb2018/masks', exist_ok=True)

    for i in tqdm(range(len(paths))):
        path = paths[i]
        img = np.array(Image.open(os.path.join(path, 'images', os.path.basename(path) + '.png')))
        mask = np.zeros((img.shape[0], img.shape[1]))

        for mask_path in glob(os.path.join(path, 'masks', '*')):
            mask_ = np.array(Image.open(mask_path)) > 0
            mask += mask_

        mask = np.where(mask > 1, 1, mask)
        mask *= 255
        img_ = Image.fromarray(img)
        img_ = img_.resize((img_size, img_size))
        mask_ = Image.fromarray(mask.astype(np.uint8))
        mask_ = mask_.resize((img_size, img_size))

        img_.save('data/dsb2018/images/' + os.path.basename(path) + '.png')
        mask_.save('data/dsb2018/masks/' + os.path.basename(path) + '.png')

        # pdb.set_trace()


if __name__ == '__main__':
    main()