import torch
import torchvision
from torchvision import datasets, transforms
from core.complexity.complexity_measures import get_all_measures
from core.model.vgg import VGG
from torch import nn
import copy
from tqdm import tqdm

def get_dataloader(selectedTask):
    if selectedTask == 'cifar-10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='./core/data', train=True, download=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=256, shuffle=False, num_workers=0)
        return train_loader
    else:
        if selectedTask == 'svhn':
            pass


def test_cnn(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    sum_loss = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            loss = criterion(output, target)
            sum_loss += len(data) * loss.item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    #test_loss /= len(test_loader.dataset)
    sum_loss /= len(test_loader.dataset)
    test_acc = correct/len(test_loader.dataset)
    return sum_loss, test_acc


def get_complexity(selectedTask):
    use_cuda = torch.cuda.is_available()
    # torch.manual_seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader = get_dataloader(selectedTask)
    criterion = nn.CrossEntropyLoss()
    model = VGG('VGG16').to(device)
    init_model = copy.deepcopy(model)
    state_dict = torch.load('./core/upload/VGG16_trloss_0.01.pt')
    model.load_state_dict(state_dict)
    trnloss, trnacc = test_cnn(model, device, train_loader, criterion)
    value = get_all_measures(model, init_model, train_loader, trnacc, seed=0)
    return value

def main():
    value = get_complexity('cifar-10')
    print(value)


if __name__ == '__main__':
    main()