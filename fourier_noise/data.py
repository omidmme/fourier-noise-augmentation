import os
import sys

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base)

import torch
import torchvision


class DatasetBuilder(object):
    # fbdb (FourierBasisDB) is original formula driven dataset generated from Fourier Basis.
    # about fbdb, please check https://github.com/gatheluck/FourierBasisDB.

    def __init__(self, **kwargs):
        """
        Args
        - name (str)       : name of dataset
        - input_size (int) : input image size
        - mean (tuple)     : mean of normalized pixel value of channels
        - std (tuple)      : standard deviation of normalized pixel value of channels
        - root_path (str)  : root path to dataset
        """
        self.root_path = 'data/cifar10'

    def __call__(self, train: bool, normalize: bool, num_samples: int = -1, binary_target: int = None,
                 corruption_type: str = None, optional_transform=[], **kwargs):
        """
        Args
        - train (bool)              : use train set or not.
        - normalize (bool)          : do normalize or not.
        - num_samples (int)         : number of samples. if -1, it means use all samples.
        - binary_target (int)       : if not None, creates datset for binary classification.
        - corruption_type (str)     : type of corruption. only avilable for cifar10c.
        - optional_transform (list) : list of optional transformations. these are applied before normalization.
        """

        transform = self._get_transform("cifar10", 32, train, normalize,
                                        optional_transform)

        # get dataset
        dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
        targets_name = 'targets'

        # make binary classification dataset
        if binary_target is not None:
            dataset = self._binarize_dataset(dataset, targets_name, binary_target)

        # take subset
        if num_samples != -1:
            assert num_samples > 0, 'num_samples should be larger than 0 or -1'
            num_samples = min(num_samples, len(dataset))
            indices = [i for i in range(num_samples)]
            dataset = torch.utils.data.Subset(dataset, indices)

        return dataset

    def _binarize_dataset(self, dataset, targets_name: str, binary_target: int):
        """
        Args
        - dataset             : pytorch dataset class.
        - targets_name (str)  : intermediate variable to compensate inconsistent torchvision API.
        - binary_target (int) : true class label of binary classification.
        """
        targets = getattr(dataset, targets_name)
        assert 0 <= binary_target <= max(targets)

        targets = [1 if target == binary_target else 0 for target in targets]
        setattr(dataset, targets_name, targets)

        return dataset

    def _get_transform(self, name: str, input_size: int, train: bool, normalize: bool, optional_transform=[]):
        """
        Args
        - name (str)                : name of dataset.
        - input_size (int)          : input image size.
        - mean (tuple)              : mean of normalized pixel value of channels
        - std (tuple)               : standard deviation of normalized pixel value of channels
        - train (bool)              : use train set or not.
        - normalize (bool)          : normalize image or not.
        - optional_transform (list) : list of optional transformations. these are applied before normalization.
        """
        transform = []

        # arugmentation
        # imagenet100 / imagenet / fbdb
        # cifar10 / cifar10c  / svhn / fbdb
        if input_size == 32:
            if train:
                transform.extend([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomCrop(32, 4),
                ])
            else:
                pass
        else:
            raise NotImplementedError

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(), ])

        # optional (Fourier Noise, Patch Shuffle, etc.)
        if optional_transform:
            transform.extend(optional_transform)

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=(0.49139968, 0.48215841, 0.44653091),
                                                 std=(0.24703223, 0.24348513, 0.26158784)),
            ])

        return torchvision.transforms.Compose(transform)
