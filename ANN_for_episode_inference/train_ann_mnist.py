import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import argparse
import utils
import torch
import matplotlib.pyplot as plt
import torch.functional as F
from model import *
from tensorboardX import SummaryWriter
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='ANN training')
parser.add_argument(
    '--root_path', default='exp_pmnist_2024/exp_gate128/', type=str)
parser.add_argument('--lr_s1', default=0.0001,
                    type=float, help='learning rate')
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--task_permutation_up', default=400, type=int)
parser.add_argument('--task_permutation_down', default=300, type=int)
parser.add_argument('--task_times', default=699, type=int)
parser.add_argument('--task_per_group', default=4, type=int)
parser.add_argument('--test_task_num', default=40, type=int)
parser.add_argument('--stage_1_percentage', default=0., type=float)

parser.add_argument('--stage_1_epoch', default=60, type=int)
parser.add_argument('--dist_regcof', default=0.05, type=float)
parser.add_argument('--l_order', default=4., type=float)
parser.add_argument('--softmax_scale', default=50., type=float)


parser.add_argument('--inputs_size', default=28*28*1, type=int)
parser.add_argument('--channels', default=256, type=int)
parser.add_argument('--n_value', default=10, type=int)
parser.add_argument('--n_dim', default=64, type=int)

parser.add_argument('--seed', default=312, type=int)
parser.add_argument('--gpu', default=1, type=int)

args = parser.parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)
s_writer = SummaryWriter(args.root_path + '/runs')


def create_folders():
    if (os.path.exists(args.root_path)):
        print('alreadly exist')
        os.system('rm -r {}'.format(args.root_path))

    os.mkdir(args.root_path)
    os.mkdir(args.root_path + '/models')
    os.mkdir(args.root_path + '/figs')


def permute_index(item):
    p_num = int(np.random.rand()*(args.task_permutation_up -
                args.task_permutation_down))+args.task_permutation_down
    start = int(np.random.rand()*(args.inputs_size-p_num))
    index = item.copy()
    result = index[:start] + np.random.permutation(
        index[start:start+p_num]).tolist()+index[start+p_num:]
    return result


def get_permutation_index(times):
    ori_index = [i for i in range(args.inputs_size)]
    ori_index = np.random.permutation(ori_index).tolist()
    index = [ori_index]
    for i in range(times):
        if ((i+1) % args.task_per_group == 0):
            temp = np.random.permutation(index[-1].copy()).tolist()
        else:
            temp = index[-1]
        index.append(permute_index(temp))
    index = np.array(index)
    print('task num:{}'.format(index.shape[0]))
    return index


def get_dist(n_value):
    return (torch.ones(n_value)/n_value).cuda()


def get_gate_net():
    net = aux_net(inputs_size=args.inputs_size, hidden_size_list=[
                  args.n_dim, args.n_dim], channels=args.channels, n_value=args.n_value).cuda()
    return net


def get_data_set():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train_ann_task(gate_net, data_set, permutation_index_list):

    def get_train_index():
        start_index = len(permutation_index_list)-args.test_task_num
        result_1 = [i for i in range(start_index)]
        result_2 = [start_index+int(args.task_per_group * group_index + inner_group_index+1)
                    for group_index in range(int(args.test_task_num/args.task_per_group))
                    for inner_group_index in range(int(args.task_per_group * args.stage_1_percentage))]
        print('stage2_index_in_stage1:', result_2)
        result = result_1+result_2
        return result

    train_index = get_train_index()
    np.save(args.root_path+'/permutation_index_for_training_ANN.npy',
            permutation_index_list[train_index, :])
    np.save(args.root_path+'/permutation_index_for_training_SNN.npy',
            permutation_index_list[-args.test_task_num:, :])

    # optimizer_aux = optim.SGD(gate_net.parameters(), lr=args.lr_s1,momentum=0.9)
    optimizer_aux = optim.Adam(
        gate_net.parameters(), lr=args.lr_s1, betas=(0.0, 0.999))

    print('stage_1:-------------------')
    for i in range(args.stage_1_epoch):
        train_ann_epoch(
            gate_net, i, data_set[0], permutation_index_list, optimizer_aux, train_index)

    test_gate = test_task(
        gate_net, data_set[1], permutation_index_list[-args.test_task_num:, :])

    return test_gate


def train_ann_epoch(gate_net, epoch, train_set, permutation_index_list, optimizer_aux, train_index):
    gate_net.train()

    expected_dist = get_dist(args.n_value)

    def adjust_lr(epoch):
        lr = args.lr_s1 * (0.1 ** (epoch // 30))
        return lr

    lr = adjust_lr(epoch)
    for p in optimizer_aux.param_groups:
        p['lr'] = lr

    print('\nEpoch: %d,lr: %.5f' % (epoch, lr))
    train_loss = 0
    loss_1_cum = 0
    loss_2_cum = 0
    total = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_set)):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.reshape(inputs.size(0), -1)

        task_index_1 = int(np.random.rand()*len(train_index))
        task_index_1 = train_index[task_index_1]

        task_index_2 = int(np.random.rand() * len(train_index))
        task_index_2 = train_index[task_index_2]

        inputs_1 = inputs[:, permutation_index_list[task_index_1]]
        inputs_2 = inputs[:, permutation_index_list[task_index_2]]

        optimizer_aux.zero_grad()
        gate_1 = gate_net(inputs_1, scale=args.softmax_scale)
        gate_2 = gate_net(inputs_2, scale=args.softmax_scale)

        regularizer_list = []
        num = len(gate_1)
        for i in range(num):
            regularizer_list.append(utils.get_conjugate_loss(
                gate_1[i], gate_2[i], permutation_index_list[task_index_1], permutation_index_list[task_index_2], l_order=args.l_order).unsqueeze(0))
        regularizer_list = torch.cat(regularizer_list, dim=0)
        loss_1 = regularizer_list.max()

        dist_loss_list = []
        for i in range(num):
            dist_loss_list.append(utils.get_dist_loss(
                gate_1[i], expected_dist).unsqueeze(0))
            dist_loss_list.append(utils.get_dist_loss(
                gate_2[i], expected_dist).unsqueeze(0))
        dist_loss_list = torch.cat(dist_loss_list, dim=0)
        loss_2 = dist_loss_list.max()

        loss = loss_1+loss_2*args.dist_regcof
        loss.backward()

        optimizer_aux.step()

        train_loss += loss.item()
        loss_1_cum += loss_1.item()
        loss_2_cum += loss_2.item()

        total += targets.size(0)
        indicator = int(len(train_set) / 3)
        if ((batch_idx + 1) % indicator == 0):
            print('total loss:{:.3f}= loss_1:{:.3f},loss_2:{:.3f}'.format(
                train_loss/(batch_idx+1), loss_1_cum/(batch_idx+1), loss_2_cum/(batch_idx+1)))

    s_writer.add_scalar('total_loss', train_loss/(batch_idx+1), epoch)
    s_writer.add_scalar('simlarity_loss', loss_1_cum/(batch_idx+1), epoch)
    s_writer.add_scalar('center_loss', loss_2_cum/(batch_idx+1), epoch)


def test_task(gate_net, test_set, permutation_index_list):
    print('testing-------------------------------------')

    def unfold(gate):
        gate = [np.concatenate(item, axis=1) for item in gate]
        gate = np.concatenate(gate, axis=0)
        return gate

    gate_list = []

    for i in tqdm(range(args.test_task_num)):
        gate = test_epoch(gate_net, test_set, permutation_index_list, i)
        temp = unfold(gate)
        temp = np.mean(temp, axis=0, keepdims=True)
        gate_list.append(temp)
    gate = np.array(gate_list)
    print('gate shape:', gate.shape)

    return gate


def test_epoch(gate_net, test_set, permutation_index_list, task_index):
    gate_net.eval()
    gate_list = []
    for batch_idx, (inputs, targets) in enumerate(test_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.reshape(inputs.size(0), -1)
        inputs = inputs[:, permutation_index_list[task_index]]
        gate = gate_net(inputs, scale=args.softmax_scale)
        gate = [item.detach().cpu().numpy() for item in gate]
        gate_list.append(gate)
    return gate_list


def print_args(args):
    dict = vars(args)
    print('arguments:--------------------------')
    with open(args.root_path+'/arguments.txt', 'w') as f:
        for key in dict.keys():
            print(key, dict[key])
            f.write(key+' '+str(dict[key])+'\n')
    print('-----------------------------------')


def main():
    print(args.root_path)
    create_folders()
    print_args(args)
    print('--------pre_task---------')
    permutation_index_list = get_permutation_index(args.task_times)
    gate_net = get_gate_net()
    data_set = get_data_set()
    print('--------train_task---------')

    modulation_signal_for_snn = train_ann_task(
        gate_net, data_set, permutation_index_list)
    np.save(args.root_path+'/modulation_signal_for_training_SNN.npy',
            modulation_signal_for_snn)

    plt.hist(modulation_signal_for_snn.flatten(), bins=200)
    plt.savefig(args.root_path+'/modulation_signal_hist.png', dpi=300)
    s_writer.add_image('hist_modulation_singal', plt.imread(
        args.root_path+'/modulation_signal_hist.png'), dataformats='HWC')

    for i in range(modulation_signal_for_snn.shape[-1]):
        data = modulation_signal_for_snn[:, :, :, i]
        data = data.reshape(data.shape[0], -1)
        data = torch.from_numpy(data).cuda()
        corr = utils.get_cosine_similarity(data).cpu().numpy()
        corr = 1-corr

        plt.matshow(corr, cmap='jet')
        plt.colorbar()
        plt.savefig(args.root_path+'/figs/corr_dim{}.png'.format(i), dpi=300)
        plt.close()

    torch.save(gate_net.state_dict(), args.root_path+'/models/gate_net.pth')


main()
