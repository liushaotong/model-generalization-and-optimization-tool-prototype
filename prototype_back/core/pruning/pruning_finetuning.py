from core.pruning.train import *
from core.pruning.utils import *
from core.pruning.models.nin_hyper import NiN
from core.pruning.models.nin_experiment_config import HParams

from core.pruning.models.gate_function import *
from torch.optim.lr_scheduler import MultiStepLR


def pruning_finetuning(pruned_state, trainloader, testloader):
    net = NiN(depth=HParams.model_depth, width=HParams.model_width, base_width=HParams.base_width,
              dataset_type=HParams.dataset_type, cfg=pruned_state['cfg'])
    net.load_state_dict(pruned_state['state_dict'])
    net.cuda()
    params = [
        {'params': net.parameters()}
    ]
    Epoch = 2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr=0.1,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5 * Epoch), int(0.75 * Epoch)], gamma=0.1)
    best_acc = 0
    best_state = {}

    best_acc, _, __, _____ = valid(0, net, testloader, best_acc, best_state, hyper_net=None)
    for epoch in range(0, Epoch):
        scheduler.step()
        ___, ____ = retrain(epoch, net, criterion, trainloader, optimizer, smooth=False, alpha=0.5)
        best_acc, valid_loss, valid_acc, best_state = valid(epoch, net, testloader, best_acc, best_state, hyper_net=None,
                                                flag=True)
    best_state['cfg'] = pruned_state['cfg']
    torch.save(best_state, './core/upload/finetuned_model.tar')
    return best_state