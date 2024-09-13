import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import argparse

import torch
import matplotlib.pyplot as plt
import torch.functional as F
from model import *
from tensorboardX import SummaryWriter
from tqdm import tqdm

import dataset_cifar100 as dataset
import utils_cifar100 as utils


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='ANN training')
parser.add_argument(
    '--root_path', default='exp/tinyimagenet2cifar100_64', type=str)
parser.add_argument('--lr_s1', default=0.0001,
                    type=float, help='learning rate')
parser.add_argument(
    '--dataset', default='tinyimagenet', type=str)
parser.add_argument('--class-num', nargs = '+', default=[200], type=int, help='the class number in dataset')
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--train_task_num', default=200, type=int)
parser.add_argument('--train_task_ratio', default=1.0, type=float)
parser.add_argument('--test_task_num', default=100, type=int)

parser.add_argument('--stage_1_epoch', default=10, type=int)
parser.add_argument('--dist_regcof', default=0.05, type=float)
parser.add_argument('--l_order', default=4., type=float)
parser.add_argument('--softmax_scale', default=50., type=float)

parser.add_argument('--inputs_size', default=768, type=int)
parser.add_argument('--channels', default=256, type=int)
parser.add_argument('--n_value', default=10, type=int)
parser.add_argument('--n_dim', default=64, type=int)

parser.add_argument('--seed', default=312, type=int)
parser.add_argument('--gpu', default=0, type=int)

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


def get_dist(n_value):
    return (torch.ones(n_value)/n_value).cuda()


def get_gate_net():
    net = aux_net(inputs_size=args.inputs_size, hidden_size_list=[
                  args.n_dim, args.n_dim], channels=args.channels, n_value=args.n_value).cuda()
    return net


def generate_task_loader(task_index_list, class_num_per_task=1, train=True):
    class_list = []
    for i in task_index_list:
        class_list.append(i * class_num_per_task +
                          np.arange(class_num_per_task))
    class_list = np.concatenate(class_list, axis=0)
    print('class_list: ', class_list)
    if (train == True):
        if (args.dataset == 'tinyimage'):
            dataset_ = dataset.Dataset(
                '../class_incremental_learning/embeddings/tinyimagenet_clip_vitl14/train', class_list=class_list)
        elif (args.dataset == 'imagenet'):
            dataset_ = dataset.Dataset(
                '../class_incremental_learning/embeddings/imagenet_clip_vitl14/train', class_list=class_list)
        elif (args.dataset == 'cifar100'):
            dataset_ = dataset.Dataset(
                '../class_incremental_learning/embeddings/cifar100_clip_vitl14/train', class_list=class_list)
        elif (args.dataset == 'imagenet+tiny'):
            dataset_ = dataset.Dataset(
                '../class_incremental_learning/embeddings/imagenet+tiny_clip_vitl14/train', class_list=class_list)
    else:
        dataset_ = dataset.Dataset(
            '../class_incremental_learning/embeddings/cifar100_clip_vitl14/test', class_list=class_list)
    data_loader = torch.utils.data.DataLoader(
        dataset_, batch_size=args.batch_size, shuffle=True)
    return data_loader


def train_ann_task(gate_net):
    if (args.dataset == 'tinyimage'):
        task_similarity_matrix = np.load(
            '../cor/tinyimagenet_vitl14.npy')
    elif (args.dataset == 'imagenet'):
        task_similarity_matrix = np.load(
            '../cor/imagenet_vitl14.npy')
    elif (args.dataset == 'cifar100'):
        task_similarity_matrix = np.load(
            '../cor/cifar100_vitl14.npy')
    elif (args.dataset == 'imagenet+tiny'):
        task_similarity_matrix = np.load(
            '../cor/imagenet+tiny_vitl14.npy')

    task_similarity_matrix = torch.from_numpy(task_similarity_matrix).cuda()
    # task_similarity_matrix[task_similarity_matrix < 0.40] = 0

    # task_similarity_matrix = torch.eye(task_similarity_matrix.shape[0]).cuda()

    # optimizer_aux = optim.SGD(gate_net.parameters(), lr=args.lr_s1,momentum=0.9)
    optimizer_aux = optim.Adam(
        gate_net.parameters(), lr=args.lr_s1, betas=(0.0, 0.999), weight_decay=0.0001)

    task_index_list = np.arange(args.train_task_num)
    # np.random.shuffle(task_index_list)
    train_task_index_list = task_index_list[:int(
        args.train_task_num*args.train_task_ratio)]

    train_set = generate_task_loader(
        train_task_index_list, train=True)

    test_task_index_list = np.arange(args.test_task_num)
    test_set = generate_task_loader(
        test_task_index_list, train=False)

    print('stage_1:-------------------')
    for i in range(args.stage_1_epoch):
        train_ann_epoch(
            gate_net, i, train_set,  optimizer_aux, task_similarity_matrix)

    test_gate = test_task(gate_net, test_set)

    return test_gate


def train_ann_epoch(gate_net, epoch, train_set,  optimizer_aux, task_similarity_matrix):
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

        # add gaussian noise
        # if (torch.rand(1).item() < 0.5):
        #     inputs = inputs + torch.randn_like(inputs)*0.10

        optimizer_aux.zero_grad()
        gate_list = gate_net(inputs, scale=args.softmax_scale)

        row_index = targets[:, None]
        col_index = targets
        sim_gt = task_similarity_matrix[row_index, col_index]

        sim_gt = 1-sim_gt

        regularizer_list = []
        num = len(gate_list)
        for i in range(num):
            regularizer_list.append(utils.get_conjugate_loss(
                gate_list[i], sim_gt, l_order=args.l_order).unsqueeze(0))

        regularizer_list = torch.cat(regularizer_list, dim=0)
        loss_1 = regularizer_list.max()

        dist_loss_list = []
        for i in range(num):
            dist_loss_list.append(utils.get_dist_loss(
                gate_list[i], expected_dist).unsqueeze(0))
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


def test_task(gate_net, test_set):
    print('testing-------------------------------------')

    def unfold(gate):
        gate = [np.concatenate(item, axis=1) for item in gate]
        gate = np.concatenate(gate, axis=0)
        return gate

    gates, targets = test_epoch(gate_net, test_set)

    targets = np.concatenate(targets, axis=0)
    index = np.argsort(targets)

    gates = unfold(gates)
    gates = gates[index]
    gates = gates.reshape(100, 100, -1, args.n_value)
    gates = np.mean(gates, axis=1)
    return gates


def test_epoch(gate_net, test_set):
    gate_net.eval()
    gates_list = []
    targets_list = []
    for batch_idx, (inputs, targets) in enumerate(test_set):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs = inputs.reshape(inputs.size(0), -1)
        gate = gate_net(inputs, scale=args.softmax_scale)
        gate = [item.detach().cpu().numpy() for item in gate]
        gates_list.append(gate)
        targets_list.append(targets.detach().cpu().numpy())
    return gates_list, targets_list


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

    gate_net = get_gate_net()

    print('--------train_task---------')

    modulation_signal_for_snn = train_ann_task(
        gate_net)
    np.save(args.root_path+'/modulation_signal_for_training_SNN.npy',
            modulation_signal_for_snn)

    plt.hist(modulation_signal_for_snn.flatten(), bins=200)
    plt.savefig(args.root_path+'/modulation_signal_hist.png', dpi=300)
    s_writer.add_image('hist_modulation_singal', plt.imread(
        args.root_path+'/modulation_signal_hist.png'), dataformats='HWC')

    for i in range(modulation_signal_for_snn.shape[-1]):
        data = modulation_signal_for_snn[:, :, i]
        data = data.reshape(data.shape[0], -1)
        data = torch.from_numpy(data).cuda()
        corr = utils.get_cosine_similarity_1dim(data).cpu().numpy()
        corr = 1-corr

        plt.matshow(corr, cmap='jet')
        plt.colorbar()
        plt.savefig(args.root_path+'/figs/corr_dim{}.png'.format(i), dpi=300)
        plt.close()

    torch.save(gate_net.state_dict(), args.root_path+'/models/gate_net.pth')


main()
