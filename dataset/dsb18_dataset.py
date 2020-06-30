import os
import glob
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from skimage import io, transform
import pandas as pd

import pdb


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, to_transform):
        image, mask = np.array(to_transform[0]), np.array(to_transform[1])

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for mask because for images,
        # x and y axes are axis 1 and 0 respectively
        # mask = mask * [new_w / w, new_h / h]
        mask = transform.resize(mask, (new_h, new_w))

        return img, mask


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, to_transform):
        image, mask = np.array(to_transform[0]), np.array(to_transform[1])

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        mask = mask[top: top + new_h, left: left + new_w]

        return image, mask


class RandomRotate(object):
    """Rotate randomly the image in a sample.

    Args:
        angle (tuple or int): Rotation angle in degrees in counter-clockwise direction.
    """

    def __init__(self, angle):
        self.max_angle = angle

    def __call__(self, to_transform):
        image, mask = np.array(to_transform[0]), np.array(to_transform[1])
        angle = np.random.randint(-self.max_angle, self.max_angle)
        image = transform.rotate(image, angle=angle, preserve_range=True)
        mask = transform.rotate(mask, angle=angle, preserve_range=True)
        return image, mask


class DSB18Dataset(Dataset):
    def __init__(self, opt):

        data_files = sorted(glob.glob(os.path.join("./data", opt.dataset_name, "images/*.*")))
        mask_files = sorted(glob.glob(os.path.join("./data", opt.dataset_name, "masks/*.*")))

        try:
            table_df = pd.read_csv(os.path.dirname(os.path.join("./data", opt.dataset_name, "images")) + '/train_test_split.csv')
            print('Using existing train/test split')
        except:
            print('Generating new train/test split')
            table_df = pd.DataFrame(list(zip(data_files, mask_files)), columns=['data_files', 'mask_files'])
            mask = np.random.rand(len(table_df)) < 0.8
            table_df['test'] = mask
            table_df.to_csv(os.path.dirname(os.path.join("./data", opt.dataset_name, "images")) + '/train_test_split.csv')

        self.test = opt.test
        if self.test:
            self.data_files = table_df[~table_df.test].data_files.tolist()
            self.mask_files = table_df[~table_df.test].mask_files.tolist()
        else:
            self.data_files = table_df[table_df.test].data_files.tolist()
            self.mask_files = table_df[table_df.test].mask_files.tolist()

        img = Image.open(self.data_files[0])
        w, h = img.size

        transfrom_list = []
        if self.test:
            transfrom_list = [transforms.Resize((int(h * opt.scale), int(w * opt.scale)), Image.BICUBIC)]
        else:
            transfrom_list = [RandomRotate(30), RandomCrop((int(h * opt.scale), int(w * opt.scale)))]

        self.transform = transforms.Compose(transfrom_list)

    def __getitem__(self, index):
        assert os.path.basename(self.data_files[index][:12]) == os.path.basename(self.mask_files[index][:12])

        img = Image.open(self.data_files[index]).convert('RGB')
        mask = Image.open(self.mask_files[index])

        if self.test:
            img_transform = self.transform(img)
            mask_transform = self.transform(mask)
        else:
            img_transform, mask_transform = self.transform([img, mask])

        # HWC to CHW
        img_transform = np.array(img_transform).transpose((2, 0, 1)) / 255
        mask_transform = np.array(mask_transform)[np.newaxis, :] / 255

        return {"input": img_transform, "gt": mask_transform, "filepath": self.data_files[index]}

    def __len__(self):
        return len(self.data_files)
