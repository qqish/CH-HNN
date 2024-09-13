import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import argparse

from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import torch.functional as F
from model import *
from utils import createHyperparametersFile
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
    '--root_path', default='try', type=str)
parser.add_argument('--lr_s1', default=0.0001,
                    type=float, help='learning rate')
parser.add_argument(
    '--dataset', nargs = '+', default=['tinyimage'], type=str)
parser.add_argument('--class-num', nargs = '+', default=[200], type=int, help='the class number in dataset')
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--train_task_num', nargs = '+', default=[150,60], type=int)
parser.add_argument('--test_task_num', nargs = '+', default=[200,100], type=int)
parser.add_argument('--task-sequence', nargs = '+', default=[0,50,100,150,200], type=int)

parser.add_argument('--stage_1_epoch', default=100, type=int)
parser.add_argument('--dist_regcof', default=0.05, type=float)
parser.add_argument('--l_order', default=4., type=float)
parser.add_argument('--softmax_scale', default=50., type=float)
parser.add_argument('--meta', type = float, nargs = '+',  default = [1.35], metavar = 'M', help='Metaplasticity coefficients layer wise')

parser.add_argument('--inputs_size', default=768, type=int)
parser.add_argument('--channels', default=256, type=int)
parser.add_argument('--n_value', default=10, type=int)
parser.add_argument('--n_dim', default=128, type=int)

parser.add_argument('--seed', default=312, type=int)
parser.add_argument('--gpu', default=5, type=int)

args = parser.parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

args.root_path = './Results/ANN_Prior/' + args.root_path
s_writer = SummaryWriter(args.root_path + '/runs')

def create_folders():
    if (os.path.exists(args.root_path)):
        print('alreadly exist')
        os.system('rm -r {}'.format(args.root_path))

    os.mkdir(args.root_path)
    os.mkdir(args.root_path + '/models')
    for i in range(len(args.task_sequence)-1):
        os.mkdir(args.root_path + '/figs_{}'.format(i))
    # os.mkdir(args.root_path + '/figs')


def get_dist(n_value):
    return (torch.ones(n_value)/n_value).cuda()


def get_gate_net():
    net = aux_net(inputs_size=args.inputs_size, hidden_size_list=[
                  args.n_dim, args.n_dim], channels=args.channels, n_value=args.n_value).cuda()
    # net.load_state_dict(torch.load('/data1/sqq/Results/ANN_of_HMN_one_hot_20231026/exp_tinyimage_2024_random200/exp_gate{}/models/gate_net.pth'.format(args.n_dim)))
    return net

from torch.utils.data import Dataset, ConcatDataset

class CustomDataset(Dataset):
    def __init__(self, dataset, label_shift=0, transform=None):
        super().__init__()
        self.transform=transform
        self.images=[]
        self.labels=[]
        for img, label in dataset:
            self.images.append(img)
            self.labels.append(label+label_shift)
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image=self.images[idx]
        label=self.labels[idx]
        if self.transform:
            image=self.transform(image)
        return image, label
    
from datetime import datetime
def plot_parameters(model, path, epoch, save=True):
    
    fig = plt.figure(figsize=(15, 10))
    i = 1
    # subplot_num=plot_num//2+1

    for (n, p) in model.named_parameters():
        # print('bias?:',n,n.find('bias'))
        if (n.find('bias') == -1) and (n.find('bn') == -1) and (len(p.size()) != 1):# and (p.size()[0]!=100):  #bias or batchnorm weight -> no plot, only plot the fc weight layer
            fig.add_subplot(2,3,i)
            weights = p.data.cpu().numpy()            #TVGG or FVGG plot p
            binet = 50
            i+=1
            plt.title( n.replace('.','_') ,fontsize=20)
            plt.hist( weights.flatten(), binet)

    time = datetime.now().strftime('%H-%M-%S')
    fig.savefig(path+'/'+time+'_'+str(epoch)+'_weight_distribution.png')
    plt.close()

def generate_task_loader(task_index_list, dataset_name, label_shift, class_num_per_task=1, train=True):
    transforms_type = ['']#,'_transform1','_transform2','_transform3','_transform4']
    class_list = []
    for i in task_index_list:
        class_list.append(i * class_num_per_task +
                          np.arange(class_num_per_task))
    class_list = np.concatenate(class_list, axis=0)
    if (train == True):
        dataset_ = dataset.Dataset(
            './embeddings/{}_clip_vitl14/train/'.format(dataset_name), class_list=class_list, transforms_type=transforms_type)
    else:
        dataset_ = dataset.Dataset(
            './embeddings/{}_clip_vitl14/test/'.format(dataset_name), class_list=class_list, transforms_type=transforms_type)
    custom_dataset = CustomDataset(dataset_, label_shift)

    # data_loader = torch.utils.data.DataLoader(
    #         dataset_, batch_size=args.batch_size, shuffle=True)
    
    return custom_dataset


def train_ann_task(gate_net, meta={}, root_path='try'):
    weight_plot=0
    task_similarity_matrix = np.load(
        './cor/vitl14_{}.npy'.format(args.dataset[0]))
    task_similarity_matrix = torch.from_numpy(task_similarity_matrix).cuda()
    label_shift_all=[0,200]
    task_sequence = args.task_sequence
    for j in range(len(args.dataset)):
        for i in range(len(task_sequence)-1):
            weight_plot+=1
            optimizer_aux = Adam_meta(gate_net.parameters(), lr=args.lr_s1, betas=(0.0, 0.999), meta=meta, )

            task_index_list_test = np.arange(args.class_num[j])
            task_index_list_train = np.arange(task_sequence[i],task_sequence[i+1])
            np.random.shuffle(task_index_list_train)
            np.random.shuffle(task_index_list_test)

            train_dataset = generate_task_loader(
                task_index_list_train, args.dataset[j], label_shift_all[j], class_num_per_task=1, train=True)
            test_dataset = generate_task_loader(
                task_index_list_test[-args.test_task_num[j]:], args.dataset[j], label_shift_all[j], class_num_per_task=1, train=False)

            data_loader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            data_loader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


            print('stage_1:-------------------')
            train_ann_epoch(gate_net, data_loader_train,  optimizer_aux, task_similarity_matrix)
            plot_parameters(gate_net, root_path, i, weight_plot)# plot hidden weights of model
            test_gate = test_task(gate_net, data_loader_test)
            np.save(root_path+'/modulation_signal_for_training_SNN_{}.npy'.format(i),
            test_gate)
            torch.save(gate_net.state_dict(), args.root_path+'/models/gate_net_{}.pth'.format(i))
            plot_ms_correlation(test_gate, i)

    return test_gate


def train_ann_epoch(gate_net, data_loader_train,  optimizer_aux, task_similarity_matrix):
    for epoch in range(args.stage_1_epoch):
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

        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader_train)):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.reshape(inputs.size(0), -1)
            # print('inputs:', inputs.shape, 'targets:', targets)
            
            #add gaussian noise 
            if(torch.rand(1).item()<0.5):
                inputs = inputs + torch.randn_like(inputs)*0.20

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
            indicator = int(len(data_loader_train) / 3)
            if ((batch_idx + 1) % indicator == 0):
                print('total loss:{:.3f}= loss_1:{:.3f},loss_2:{:.3f}'.format(
                    train_loss/(batch_idx+1), loss_1_cum/(batch_idx+1), loss_2_cum/(batch_idx+1)))

        s_writer.add_scalar('total_loss', train_loss/(batch_idx+1), epoch)
        s_writer.add_scalar('simlarity_loss', loss_1_cum/(batch_idx+1), epoch)
        s_writer.add_scalar('center_loss', loss_2_cum/(batch_idx+1), epoch)


def test_task(gate_net, data_loader_test):
    print('testing-------------------------------------')

    def unfold(gate):
        gate = [np.concatenate(item, axis=1) for item in gate]
        gate = np.concatenate(gate, axis=0)
        return gate

    gates, targets = test_epoch(gate_net, data_loader_test)

    targets = np.concatenate(targets, axis=0)
    index = np.argsort(targets)

    gates = unfold(gates)
    gates = gates[index]
    print('gates:', gates.shape)
    if len(gates)==20000:
        gates_0 = gates[:10000]
        gates_1 = gates[10000:]
        gates_0 = gates_0.reshape(200, 50, -1, args.n_value)
        gates_1 = gates_1.reshape(100, 100, -1, args.n_value)
        gates_0 = np.mean(gates_0, axis=1)
        gates_1 = np.mean(gates_1, axis=1)
        gates = np.concatenate((gates_0, gates_1), axis=0)
    else:
        if args.dataset[0]=="tinyimage":
            gates = gates.reshape(200, 50, -1, args.n_value)
        elif args.dataset[0]=="cifar100":
            gates = gates.reshape(100, 100, -1, args.n_value)
        gates = np.mean(gates, axis=1)
    print('gates:', gates.shape)
    return gates


def test_epoch(gate_net, data_loader_test):
    gate_net.eval()
    gates_list = []
    targets_list = []
    print('dataloader:', len(data_loader_test))
    for batch_idx, (inputs, targets) in enumerate(data_loader_test):
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

def plot_ms_correlation(modulation_signal_for_snn,idx):
    for i in range(modulation_signal_for_snn.shape[-1]):
        data = modulation_signal_for_snn[:, :, i]
        data = data.reshape(data.shape[0], -1)
        data = torch.from_numpy(data).cuda()
        corr = utils.get_cosine_similarity_1dim(data).cpu().numpy()
        corr = 1-corr

        plt.matshow(corr, cmap='jet')
        plt.colorbar()
        plt.savefig(args.root_path+'/figs_{1}/corr_dim{0}.png'.format(i,idx), dpi=300)
        plt.close()

def main():
    print(args.root_path)
    create_folders()
    print_args(args)
    print('--------pre_task---------')

    gate_net = get_gate_net()
    # for n,p in (gate_net.named_parameters()):
    #     print(n,p.shape)
    print('--------train_task---------')
    meta = {}
    for n, p in gate_net.named_parameters():
        if ('l'in n):
            index = int(n[7])
        elif ('h' in n):
            index = int(n[6])+3
        p.newname = 'l'+str(index)
        if ('l' in n) or ('h' in n):
            meta[p.newname] = args.meta[index-1] if len(args.meta)>1 else args.meta[0]
    modulation_signal_for_snn = train_ann_task(
        gate_net, meta, args.root_path)
    createHyperparametersFile(args)
    plt.hist(modulation_signal_for_snn.flatten(), bins=200)
    plt.savefig(args.root_path+'/modulation_signal_hist.png', dpi=300)
    s_writer.add_image('hist_modulation_singal', plt.imread(
        args.root_path+'/modulation_signal_hist.png'), dataformats='HWC')

main()
