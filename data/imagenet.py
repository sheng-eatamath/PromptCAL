import torchvision
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
# from data_utils import subsample_instances
from config import imagenet_root, imagenet_gcd_root

IN_SPLIT = ['n01443537', 'n01537544', 'n01631663', 'n01644373', 'n01692333', 
            'n01729977', 'n01775062', 'n01873310', 'n01914609', 'n02028035', 
            'n02033041', 'n02091635', 'n02097047', 'n02098105', 'n02105855', 
            'n02106030', 'n02107142', 'n02107683', 'n02109525', 'n02110341', 
            'n02110627', 'n02112350', 'n02112706', 'n02113186', 'n02113799', 
            'n02114548', 'n02114855', 'n02120079', 'n02133161', 'n02137549', 
            'n02138441', 'n02174001', 'n02219486', 'n02226429', 'n02256656', 
            'n02268443', 'n02326432', 'n02480855', 'n02481823', 'n02504458', 
            'n02514041', 'n02704792', 'n02747177', 'n02749479', 'n02804610', 
            'n02869837', 'n02879718', 'n02978881', 'n02988304', 'n03017168', 
            'n03026506', 'n03028079', 'n03045698', 'n03197337', 'n03337140', 
            'n03372029', 'n03404251', 'n03417042', 'n03447447', 'n03450230', 
            'n03461385', 'n03481172', 'n03534580', 'n03617480', 'n03706229', 
            'n03710637', 'n03724870', 'n03729826', 'n03769881', 'n03792972', 
            'n03873416', 'n03877845', 'n03899768', 'n03908714', 'n03982430', 
            'n03991062', 'n03995372', 'n04070727', 'n04153751', 'n04154565', 
            'n04200800', 'n04204238', 'n04229816', 'n04296562', 'n04317175', 
            'n04442312', 'n04456115', 'n04487081', 'n04522168', 'n04591157', 
            'n04596742', 'n06785654', 'n07579787', 'n07590611', 'n07768694', 
            'n09229709', 'n10148035', 'n12144580', 'n13037406', 'n13052670']

SPLIT_FILE = [
    '/home/sheng/orca/data/ImageNet100_label_50_0.5.txt',
    '/home/sheng/orca/data/ImageNet100_unlabel_50_0.5.txt',
]

def get_data_root(data_root):
    if data_root=='imagenet_100':
        return imagenet_root
    elif data_root=='imagenet_100_gcd':
        return imagenet_gcd_root
    elif data_root=='imagenet_original_100':
        return imagenet_original_root
    else:
        raise ValueError(f'data_root={data_root}')
    return

def read_split_file():
    global SPLIT_FILE
    with open(SPLIT_FILE[0], 'r') as f:
        xl = list(filter(lambda x: len(x), f.read().split('\n')))
        xl = list(map(lambda x: x.split(' ')[0].split('/')[1], xl))
    with open(SPLIT_FILE[1], 'r') as f:
        xu = list(filter(lambda x: len(x), f.read().split('\n')))
        xu = list(map(lambda x: x.split(' ')[0].split('/')[1], xu))
    return set(xl), set(xu)

class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

def subsample_dataset(dataset, idxs, absolute=True):
    mask = np.zeros(len(dataset)).astype('bool')
    if absolute==True:
        mask[idxs] = True
    else:
        idxs = set(idxs)
        mask = np.array([i in idxs for i in dataset.uq_idxs])

    dataset.samples = [s for m, s in zip(mask, dataset.samples) if m==True]
    dataset.targets = [t for m, t in zip(mask, dataset.targets) if m==True]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):
    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs



def get_equal_len_datasets(dataset1, dataset2):
    """
    Make two datasets the same length
    """

    if len(dataset1) > len(dataset2):

        rand_idxs = np.random.choice(range(len(dataset1)), size=(len(dataset2, )))
        subsample_dataset(dataset1, rand_idxs)

    elif len(dataset2) > len(dataset1):

        rand_idxs = np.random.choice(range(len(dataset2)), size=(len(dataset1, )))
        subsample_dataset(dataset2, rand_idxs)

    return dataset1, dataset2



def get_imagenet_100_gcd_datasets(train_transform, test_transform, train_classes=range(50),
                           prop_train_labels=0.5, split_train_val=False, seed=0):

    np.random.seed(seed)

    ### >>>
    subsampled_100_classes = np.arange(100)
    ### <<<
    
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'train'), transform=train_transform)
    # NOTE: use GCD paper IN split
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None
    

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets



def get_imagenet_100_gcd_datasets_with_gcdval(train_transform, test_transform, train_classes=range(50), 
                                   prop_train_labels=0.5, split_train_val=True, seed=0, val_split=0.1):

    np.random.seed(seed)

    ### >>>
    subsampled_100_classes = np.arange(100)
    ### <<<
    
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'train'), transform=train_transform)
    # NOTE: use GCD paper IN split
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)
    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)), absolute=False)
    
    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled, val_split=val_split)
    val_dataset_labelled = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled.transform = test_transform
    train_dataset_labelled = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    
    train_idxs, val_idxs = get_train_val_indices(train_dataset_unlabelled, val_split=val_split)
    val_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset_unlabelled), val_idxs)
    val_dataset_unlabelled.transform = test_transform
    train_dataset_unlabelled = subsample_dataset(deepcopy(train_dataset_unlabelled), train_idxs)
    
    val_dataset_unlabelled.transform = test_transform
    

    # Get test set for all classes
    test_dataset = ImageNetBase(root=os.path.join(imagenet_gcd_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    print(f'total={len(whole_training_set)} train={len(train_dataset_labelled)} {len(train_dataset_unlabelled)} val={len(val_dataset_labelled)} {len(val_dataset_unlabelled)} test={len(test_dataset)}')

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': [val_dataset_labelled, val_dataset_unlabelled],
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_imagenet_100_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
    
    # imagenet_original_root = r'/home/sheng/dataset/imagenet-100-original'
    # train_dataset_labelled = ImageNetDataset(root=os.path.join(imagenet_original_root, 'train'), anno_file=SPLIT_FILE[0], transform=None)
    # print(list(set(train_dataset_labelled.targets)))
    # train_dataset_unlabelled = ImageNetDataset(root=os.path.join(imagenet_original_root, 'train'), anno_file=SPLIT_FILE[1], transform=None, start_idx=len(train_dataset_labelled))
    # print(list(set(train_dataset_unlabelled.targets)))