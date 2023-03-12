from core.pruning.train import *
from core.pruning.utils import *
from core.pruning.models.nin_hyper import NiN as NiN_hyper
from core.pruning.models.hypernet import Simplified_Gate, PP_Net, Episodic_mem, Simple_PN
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

def pruning_pm(net, validloader, testloader):
    net.eval()
    # 获取 size_out:层输出的长乘宽 size_kernel：卷积核相乘 size_group? size_inchannel：输入通道数 size_outchannel：输出通道数 ****
    size_out, size_kernel, size_group, size_inchannel, size_outchannel = get_middle_Fsize_nin(net)
    # 输出结构
    width, structure = net.count_structure()  # width：1400 strucutre：[350,350,350,350]
    # 输出一个可微的结构向量
    hyper_net = Simplified_Gate(structure=structure, T=0.4, base=3.0)
    # 性能预测网络:PP_Net门控循环，Simple_PN线性回归
    pp_net = PP_Net(structure=structure)
    net.foreze_weights()
    # 结构大小损失 ***** 这里不对
    resource_reg = Flops_constraint_nin(0.5, size_kernel, size_out, size_group, size_inchannel, size_outchannel,
                                        HN=True, structure=structure)
    criterion = nn.CrossEntropyLoss()
    Epoch = 200
    epm_flag = True
    nf = 3.0
    orth_grad = True
    sampling = True
    hyper_net.cuda()
    pp_net.cuda()
    net.cuda()
    optimizer_p = optim.AdamW(pp_net.parameters(), lr=1e-3, weight_decay=1e-3)
    optimizer = optim.AdamW(hyper_net.parameters(), lr=5e-2, weight_decay=1e-2)
    # 记忆模块
    ep_mem = Episodic_mem(K=500, avg_len=2, structure=structure, )
    scheduler = MultiStepLR(optimizer, milestones=[int(Epoch * 0.8)], gamma=0.1)
    best_acc = 0
    best_state = {}
    valid(0, net, testloader, best_acc, best_state, hyper_net=None, stage='valid_model', flag=False)
    for epoch in range(Epoch):
        train_epm(validloader, net, optimizer, optimizer_p, epoch, epm_flag, nf,
                  resource_constraint=resource_reg,
                  hyper_net=hyper_net,
                  pp_net=pp_net, epm=ep_mem, ep_bn=64, orth_grad=orth_grad, use_sampler=sampling, )

        scheduler.step()
        best_acc, valid_loss, valid_acc, best_state = valid(epoch, net, testloader, best_acc, best_state, hyper_net=hyper_net,
                                                stage='valid_model', flag=True)
    return best_state
