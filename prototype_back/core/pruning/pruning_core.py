import torch
import torchvision
import torchvision.transforms as transforms

from core.pruning.utils import *

from core.pruning.models.nin_hyper import NiN as NiN_hyper
from core.pruning.models.nin_experiment_config import HParams
from core.pruning.pruning_pm import pruning_pm
from core.pruning.pruning_model import pruning_model
from core.pruning.pruning_finetuning import pruning_finetuning


def get_dataloader(selectedTask):
    if selectedTask == 'cifar-10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./core/pruning/datasets/cifar10/', train=True, download=False,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./core/pruning/datasets/cifar10/', train=False, download=True,
                                               transform=transform_test)
        train_sampler, val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=0, shuffle=True)
        validloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=0, sampler=val_sampler)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
        return trainloader, validloader, testloader


    else:
        if selectedTask == 'svhn':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            trainset = torchvision.datasets.CIFAR10(root='./datasets/svhn/', train=True, download=False,
                                                transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./datasets/svhn/', train=False, download=True,
                                                   transform=transform_test)
            train_sampler, val_sampler = TrainVal_split(trainset, 0.1, shuffle_dataset=True)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=0, shuffle=True)
            validloader = torch.utils.data.DataLoader(trainset, batch_size=256, num_workers=0, sampler=val_sampler)
            testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
            return trainloader, validloader, testloader


def get_pruning(selectedTask):
    trainloader, validloader, testloader = get_dataloader(selectedTask)
    net = NiN_hyper(depth=HParams.model_depth, width=HParams.model_width, base_width=HParams.base_width,
                    dataset_type=HParams.dataset_type)
    state_dict = torch.load('./core/pruning/checkpoint/nin-base.pth.tar')
    net.load_state_dict(state_dict['net'])
    pruning_state = pruning_pm(net, validloader, testloader)
    pruned_state = pruning_model(pruning_state)
    finetuing_state = pruning_finetuning(pruned_state, trainloader, testloader)
    value = {'原始模型准确率（%）': state_dict['acc'], '剪枝率': pruned_state['pruning_rate'], '子模型准确率（%）': finetuing_state['acc']}
    return value


if __name__ == '__main__':
    value = get_pruning('cifar-10')
    print(value)
