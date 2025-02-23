from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np


class RotatedMNISTDataset(Dataset):
    """
    Dataset definition for the Rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(52, 52)
        digit = digit[..., np.newaxis]

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, :]
        # changed
        # rotation, shift_x, shift_y, cost
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([1, 2, 3, 4])])))

        if self.transform:
            digit = self.transform(digit)
        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample

class RotatedMNISTDataset_partial(Dataset):
    """
    Dataset definition for the Rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, label_mask_file, root_dir, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        self.label_source = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.label_mask_source = pd.read_csv(os.path.join(root_dir, label_mask_file), header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(52, 52)
        digit = digit[..., np.newaxis]

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')
#        print(self.label_mask_source.iloc[idx, :])
        label_mask = self.label_mask_source.iloc[idx, :]
        label_mask = np.array(label_mask, dtype='uint8')
        label_mask = label_mask[[1, 2, 3, 4]]
#        print(label_mask.shape)

        label = self.label_source.iloc[idx, :]
        # changed
        # rotation, shift_x, shift_y, cost
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([1, 2, 3, 4])])))

        if self.transform:
            digit = self.transform(digit)
        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask, 'label_mask': label_mask}
        return sample


class RotatedMNISTDataset_BO(Dataset):
    """
    Dataset definition for the Rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, data, train_x, mask, transform=None):

        self.data_source = data
        self.mask_source = train_x
        self.label_source = mask
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(52, 52)
        digit = digit[..., np.newaxis]

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([1, 2, 3, 4])])))
        if self.transform:
            digit = self.transform(digit)
        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDataset_partial_BO(Dataset):
    """
    Dataset definition for the Rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, data, mask, label, label_mask, transform=None):

        self.data_source = data
        self.mask_source = mask
        self.label_source = label
        self.label_mask_source = label_mask
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(52, 52)
        digit = digit[..., np.newaxis]

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label_mask = self.label_mask_source[idx, :]
        label_mask = np.array(label_mask, dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.nan_to_num(np.array(label[np.array([0, 1, 2, 3])])))
        if self.transform:
            digit = self.transform(digit)
        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask, 'label_mask': label_mask}
        return sample