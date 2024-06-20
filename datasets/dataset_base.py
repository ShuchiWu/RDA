import numpy as np

from torch.utils.data import Dataset
from PIL import Image



class DATASETBASE(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]


class DATAPAIR(DATASETBASE):

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class TESTDATA(DATASETBASE):

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target



