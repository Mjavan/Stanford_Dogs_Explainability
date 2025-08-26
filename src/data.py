import os
import abc
from abc import  abstractmethod

import numpy as np
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})

########
PATH_TO_DOGS = '../data/Images/'
########

class TestData(ABC):
    '''(Abstract) base class for all data modules
    
    Every subclass needs to implement a .get_data(H0) method, that takes a
    boolean (whether to sample from H0 or H1) and outputs two sets of observations
    from the two distributions to be tested. Output should be two np.ndarray's of
    size (m, *d) where m = #observations/population, d = dimensionality (might be
    non-scalar, e.g. d = (3, 128, 128) for images, d = 10 for 10-dim features)
    Subclasses also need to implement a .test_h0() function that returns a bool
    whether the null hypothesis at a given self.m-value can be evaluated
    '''
    def __init__(self):
        pass

    @abstractmethod
    def test_h0(self):
        pass

    @abstractmethod
    def get_data(self, H0=True):
        pass

class TorchData(TestData):

    '''(Abstract) superclass for torch data reading utility

    Subclasses only need to load data as torch.Tensors into self.c0_data
    and self.c1_data

    # Parameters:
    m (int): number of observations per sample
    '''
    def __init__(self, m=200):
        super(TorchData, self).__init__()
        self.m = m

    def test_h0(self):
        return len(self.c0_data) >= 2*self.m

    def get_data(self, H0=True):
        perm0 = torch.randperm(len(self.c0_data))
        X = self.c0_data[perm0[:self.m]]
        if H0:
            Y = self.c0_data[perm0[self.m:(2*self.m)]]
        else:
            perm1 = torch.randperm(len(self.c1_data))
            Y = self.c1_data[perm1[:self.m]]
        return np.array(X), np.array(Y)


class ImageData(TorchData):

    '''Data object for the natural image experiments

    Data must be ordered into directories named after classes.

    # Parameters:
    c0 (str): name of X-class (must match folder name)
    c1 (str): name of Y-class (must match folder name)
    path_to_data (str): path to root directory of image data; must contain directories with
                        classes c0 and c1
    m (int): number of observations per sample
    gray (bool): whether to transform the data to grayscale
    target_shape (tuple(int, int)): target shape for the images
    cropping (None or tuple(int, int)): apply center cropping to these dimensions before reshaping
    '''
    def __init__(self, c0, c1, path_to_data, m=50, gray=False, target_shape=(224, 224), cropping=None):
        super(ImageData, self).__init__(m=m)
        self.target_shape = target_shape
        self.path = path_to_data
        self.gray = gray
        self.cropping = cropping
        self.load_classes(c0, c1, path_to_data)

    def get_transform(self):
        tfms = [
            transforms.Resize(self.target_shape),
            transforms.ToTensor(),
            ]

        if self.gray:
            tfms.insert(1, transforms.Grayscale())
        else:
            tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        if self.cropping is not None:
            tfms.insert(0, transforms.CenterCrop(self.cropping))

        transform = transforms.Compose(tfms)
        return transform
            
    def load_classes(self, c0, c1, path_to_data):
        transform = self.get_transform()

        tset = torchvision.datasets.ImageFolder(path_to_data, transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=200, shuffle=True, num_workers=8)

        all_class_c0 = []
        all_class_c1 = []
        c0 = tset.classes.index(c0)
        c1 = tset.classes.index(c1)
        for data, target in tqdm(loader):
            all_class_c0.append(data[target==c0])
            all_class_c1.append(data[target==c1])
        self.c0_data = torch.cat(all_class_c0)
        self.c1_data = torch.cat(all_class_c1)

if __name__=="__main__":
    c0, c1 = 'n02090721-Irish_wolfhound', 'n02092002-Scottish_deerhound'
    data = ImageData(path_to_data=PATH_TO_DOGS, c0=c0, c1=c1, target_shape=(224, 224))

    print(type(data))

    print(data.m)
    print(data.c0_data.shape) 
    print(data.c1_data.shape) 

    print(type(data.c0_data))
    print(type(data.c1_data))

    print(data.c0_data[:50].shape)

    gr1 = data.c0_data[:50]
    gr2 = data.c1_data[:50]

    #gr1_loader = DataLoader(gr1, batch_size=5, shuffle=False, drop_last=True)

    #for img in gr1_loader:
    #    print(type(img))
    #    print(f'img:{img.shape}')
    #    break

    

