from typing import Any, Tuple
import torchvision
from torchvision import datasets,transforms
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader , WeightedRandomSampler
import os
from utils import get_mean_std , plot_data

def creat_dataset(path_train , path_test  , path_val , BATCH_SIZE=8 , img_size=150 , imbalance = False):
    transform_train = transforms.Compose([
        transforms.Resize([img_size,img_size]),
        # transforms.ColorJitter(brightness = 0.5),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomRotation(degrees=40),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4797, 0.4727, 0.4679], std=[0.2607, 0.2581, 0.2583]),
    ])

    transform_test = transforms.Compose([transforms.Resize([img_size,img_size]),
                                         transforms.ToTensor(),
                                         ])
    
    transform_val = transforms.Compose([transforms.Resize([img_size,img_size]),
                                         transforms.ToTensor(),
                                         ])
    
    train_dataset = ImageFolder(root=path_train , transform=transform_train)
    test_dataset = ImageFolder(root=path_test , transform=transform_test )
    val_dataset = ImageFolder(root=path_val , transform=transform_val)

    if imbalance:
        classes,counts = torch.unique(torch.tensor(train_dataset.targets),return_counts=True)
        weights = 1.0 / counts.float()
        sample_weights = weights[torch.tensor(train_dataset.targets).squeeze().long()]
        sampler = WeightedRandomSampler(
            weights = sample_weights,
            num_samples = len(sample_weights),
            replacement=True
        )
        trainset = DataLoader(train_dataset, batch_size=BATCH_SIZE,sampler=sampler)
        testset = DataLoader(test_dataset,batch_size=BATCH_SIZE)
        valset = DataLoader(val_dataset , batch_size=BATCH_SIZE)


        return trainset, testset ,valset
    

    trainset = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testset = DataLoader(test_dataset,batch_size=BATCH_SIZE)
    valset = DataLoader(val_dataset , batch_size=BATCH_SIZE)

    return trainset, testset , valset


if '__main__' == __name__:
    def test():
        traindata , testdata ,valdata = creat_dataset(path_train=r"cmc\train" , path_test=r"cmc\test" , path_val=r"cmc\val" , imbalance=False)
        imageT , _ = next(iter(traindata))
        print(imageT.shape)
        # plot_data(traindata)
        
        # mean , std = get_mean_std(trainset)
        # print("mean :" , mean , "std :" , std)
    test()