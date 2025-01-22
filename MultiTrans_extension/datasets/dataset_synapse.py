import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Augmentation

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)   
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

# flip、rotate、crop
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)   # flip
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)    # rotate
        
        x, y = image.shape # 2D slices

        # image resize
        if x != self.output_size[0] or y != self.output_size[1]:
            # order: The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # image (numpy format) ---> np.float32 ---> torch Tensor ---> add the batch dimension
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # x, y ---> 1, x, y
        label = torch.from_numpy(label.astype(np.float32))

        sample = {'image': image, 'label': label.long()}  # 
        return sample

# -----------------------------------------------------------------------------
# dataset
class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()  # name list
        self.data_dir = base_dir  # folder name

    def __len__(self):
        return len(self.sample_list)  # number of samples

    def __getitem__(self, idx):

        # train dataset: img, label the .npz form
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')  # slice name
            data_path = os.path.join(self.data_dir, slice_name+'.npz')

            data = np.load(data_path)  # read .npz 
            image, label = data['image'], data['label']    # from .npz extract slice of img, label

        else:
        # read val dataset' 3D img, label. .npy.h5 form
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name) 

            data = h5py.File(filepath)  # read .npy.h5 

            image, label = data['image'][:], data['label'][:]  # obtain slice sequences

        sample = {'image': image, 'label': label}

        # use the transform function of pytorch
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')  # original name

        return sample
