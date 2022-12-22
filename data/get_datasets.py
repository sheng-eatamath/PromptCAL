from data.data_utils import MergedDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.herbarium_19 import get_herbarium_datasets
from data.stanford_cars import get_scars_datasets
from data.imagenet import get_imagenet_100_gcd_datasets, get_imagenet_100_gcd_datasets_with_gcdval
from data.cub import get_cub_datasets, get_cub_datasets_with_gcdval
from data.fgvc_aircraft import get_aircraft_datasets

from data.cifar import subsample_classes as subsample_dataset_cifar
from data.herbarium_19 import subsample_classes as subsample_dataset_herb
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_air
### >>>
from torchvision.datasets import ImageNet

from copy import deepcopy
import pickle
import os

from functools import partial

import numpy as np

from config import osr_split_dir

sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'cifar100_10': subsample_dataset_cifar,
    'cifar100_25': subsample_dataset_cifar,
    'cifar100_50': subsample_dataset_cifar,
    'imagenet_100_gcd': subsample_dataset_imagenet,
    'herbarium_19': subsample_dataset_herb,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_air,
    'scars': subsample_dataset_scars
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'cifar100_10': get_cifar_100_datasets,
    'cifar100_25': get_cifar_100_datasets,
    'cifar100_50': get_cifar_100_datasets,
    'imagenet_100_gcd': get_imagenet_100_gcd_datasets,
    'herbarium_19': get_herbarium_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
}



from data.cifar import get_cifar_100_datasets_with_gcdval
from data.cub import get_cub_datasets_with_gcdval
from data.imagenet import get_imagenet_100_gcd_datasets_with_gcdval
get_dataset_funcs_with_gcdval = {
    'cifar100': get_cifar_100_datasets_with_gcdval,
    'cub': get_cub_datasets_with_gcdval,
    'imagenet_100_gcd': get_imagenet_100_gcd_datasets_with_gcdval,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=False)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


### >>>
def get_datasets_with_gcdval(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs_with_gcdval[dataset_name]
    datasets = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                            train_classes=args.train_classes,
                            prop_train_labels=args.prop_train_labels,
                            split_train_val=True, val_split=args.val_split)

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            if isinstance(dataset, list):
                for d in dataset:
                    d.target_transform = target_transform
            else:
                dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))
    val_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['val'][0]),
                                  unlabelled_dataset=deepcopy(datasets['val'][1]))
    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, val_dataset, datasets
### <<<

def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)
        
    elif args.dataset_name == 'cifar100_10':
        np.random.seed(0)
        args.image_size = 32
        ### use random generated classes
        # args.train_classes = np.random.choice(np.arange(100), size=10, replace=False).tolist()
        args.train_classes = [18, 49, 67, 16, 72, 14, 39, 47, 35, 88]
        args.unlabeled_classes = list(set(range(100))-set(args.train_classes))
        
    elif args.dataset_name == 'cifar100_25':
        np.random.seed(0)
        args.image_size = 32
        args.train_classes = np.random.choice(np.arange(100), size=25, replace=False).tolist()
        args.unlabeled_classes = list(set(range(100))-set(args.train_classes))
        
    elif args.dataset_name == 'cifar100_50':
        np.random.seed(0)
        args.image_size = 32
        args.train_classes = np.random.choice(np.arange(100), size=50, replace=False).tolist()
        args.unlabeled_classes = list(set(range(100))-set(args.train_classes))
    

    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']
        
    elif args.dataset_name == 'imagenet_100_gcd':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:
            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)
    else:

        raise NotImplementedError

    return args

