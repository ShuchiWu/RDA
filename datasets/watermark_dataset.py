import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np
import torch
import random
import copy


class TestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']


        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])
        img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img_backdoor =self.test_transform(Image.fromarray(img))
        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]