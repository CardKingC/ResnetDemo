""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
from PIL import Image
import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets import DatasetFolder
from conf import global_settings as gs
from sklearn.model_selection import KFold



def get_network(net,useCli=True):
    """ return given network
    """
    if net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    # elif net == 'inceptionv4':
    #     from models.inceptionv4 import inceptionv4
    #     net = inceptionv4()
    # elif net == 'inceptionresnetv2':
    #     from models.inceptionv4 import inception_resnet_v2
    #     net = inception_resnet_v2()
    # elif net == 'xception':
    #     from models.xception import xception
    #     net = xception()
    elif net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    # 加入临床数据的模型
    elif net == 'cresnet18':
        from models.resnet2 import resnet18
        net = resnet18(useCli=useCli)
    elif net == 'cresnet34':
        from models.resnet2 import resnet34
        net = resnet34(useCli=useCli)
    elif net == 'cresnet50':
        from models.resnet2 import resnet50
        net = resnet50(useCli=useCli)
    elif net == 'cresnet101':
        from models.resnet2 import resnet101
        net = resnet101(useCli=useCli)
    elif net == 'resnet152':
        from models.resnet2 import resnet152
        net = resnet152(useCli=useCli)
    # elif net == 'preactresnet18':
    #     from models.preactresnet import preactresnet18
    #     net = preactresnet18()
    # elif net == 'preactresnet34':
    #     from models.preactresnet import preactresnet34
    #     net = preactresnet34()
    # elif net == 'preactresnet50':
    #     from models.preactresnet import preactresnet50
    #     net = preactresnet50()
    # elif net == 'preactresnet101':
    #     from models.preactresnet import preactresnet101
    #     net = preactresnet101()
    # elif net == 'preactresnet152':
    #     from models.preactresnet import preactresnet152
    #     net = preactresnet152()
    # elif net == 'resnext50':
    #     from models.resnext import resnext50
    #     net = resnext50()
    # elif net == 'resnext101':
    #     from models.resnext import resnext101
    #     net = resnext101()
    # elif net == 'resnext152':
    #     from models.resnext import resnext152
    #     net = resnext152()
    # elif net == 'shufflenet':
    #     from models.shufflenet import shufflenet
    #     net = shufflenet()
    # elif net == 'shufflenetv2':
    #     from models.shufflenetv2 import shufflenetv2
    #     net = shufflenetv2()
    # elif net == 'squeezenet':
    #     from models.squeezenet import squeezenet
    #     net = squeezenet()
    # elif net == 'mobilenet':
    #     from models.mobilenet import mobilenet
    #     net = mobilenet()
    # elif net == 'mobilenetv2':
    #     from models.mobilenetv2 import mobilenetv2
    #     net = mobilenetv2()
    # elif net == 'nasnet':
    #     from models.nasnet import nasnet
    #     net = nasnet()
    elif net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif net == 'attention92':
        from models.attention import attention92
        net = attention92()
    # elif net == 'seresnet18':
    #     from models.senet import seresnet18
    #     net = seresnet18()
    # elif net == 'seresnet34':
    #     from models.senet import seresnet34
    #     net = seresnet34()
    # elif net == 'seresnet50':
    #     from models.senet import seresnet50
    #     net = seresnet50()
    # elif net == 'seresnet101':
    #     from models.senet import seresnet101
    #     net = seresnet101()
    # elif net == 'seresnet152':
    #     from models.senet import seresnet152
    #     net = seresnet152()
    # elif net == 'wideresnet':
    #     from models.wideresidual import wideresnet
    #     net = wideresnet()
    # elif net == 'stochasticdepth18':
    #     from models.stochasticdepth import stochastic_depth_resnet18
    #     net = stochastic_depth_resnet18()
    # elif net == 'stochasticdepth34':
    #     from models.stochasticdepth import stochastic_depth_resnet34
    #     net = stochastic_depth_resnet34()
    # elif net == 'stochasticdepth50':
    #     from models.stochasticdepth import stochastic_depth_resnet50
    #     net = stochastic_depth_resnet50()
    # elif net == 'stochasticdepth101':
    #     from models.stochasticdepth import stochastic_depth_resnet101
    #     net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net

class ImageClinicalDataset(DatasetFolder):
    '''
        加载图像和临床数据的numpy数组
    '''
    def __init__(self,root,loader,extensions=None,transform=None,target_transform=None,is_valid_file=None):
        DatasetFolder.__init__(self,root,loader,extensions,transform,target_transform,is_valid_file)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample['image'] = self.transform(Image.fromarray(sample['image']))
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        #image 已经在transform中转换为了Tensor，这里仅仅改变类型为Float
        sample['image']=sample['image'].float()
        sample['cdata']=torch.from_numpy(sample['cdata']).float()
        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def loader(x):
    data=np.load(x)
    return {'image':data['image'],
            'cdata':data['cdata']}

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])


    train_set=DatasetFolder(gs.TRAIN_DATASET_PATH,loader=lambda x:Image.open(x),extensions='png',transform=transform_train)

    #使用自定义数据加载器加载npz文件
    # train_set = ImageClinicalDataset(gs.TRAIN_DATASET_PATH, loader=loader, extensions='npz',
    #                           transform=transform_train)
    train_loader=DataLoader(train_set,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size)
    return train_loader
def get_valid_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_valid = transforms.Compose([
        #transforms.ToPILImage(),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    # cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # training_loader = DataLoader(
    #     cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    #valid_set=DatasetFolder(gs.VALID_DATASET_PATH,loader=lambda x:Image.open(x),extensions='png',transform=transform_train)

    #使用自定义数据加载器加载npz文件
    valid_set = ImageClinicalDataset(gs.VALID_DATASET_PATH, loader=loader, extensions='npz',
                              transform=transform_valid)
    valid_loader=DataLoader(valid_set,shuffle=shuffle,num_workers=num_workers,batch_size=batch_size)
    return valid_loader
def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    # cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # test_loader = DataLoader(
    #     cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    test_set = DatasetFolder(gs.TEST_DATASET_PATH, loader=lambda x: Image.open(x), extensions='png',
                               transform=transform_test)
    #使用自定义数据加载器
    # test_set = ImageClinicalDataset(gs.TEST_DATASET_PATH, loader=loader, extensions='npz',
    #                                                     transform=transform_test)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader

def get_kfolder_dataloader(k,batch_size=gs.BATCH_SIZE, num_workers=0, shuffle=True):
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])

    kf=KFold(n_splits=k,shuffle=True,random_state=0)
    data=ImageClinicalDataset(gs.TRAIN_DATASET_PATH, loader=loader, extensions='npz',
                                                        transform=transform_test)
    for train_index,val_index in kf.split(data):
        train_fold=torch.utils.data.dataset.Subset(data, train_index)
        val_fold = torch.utils.data.dataset.Subset(data, val_index)
        train_loader = DataLoader(train_fold, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        val_loader = DataLoader(val_fold, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        yield (train_loader,val_loader)

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]