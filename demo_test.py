import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Test the Performance of Various Datasets.')


parser.add_argument('--dataset', type = str, default = 'cifar100', metavar = 'Dataset', help='Dataset Defination')
parser.add_argument('--scenario', type = str, default = 'class-incre', metavar = 'Scenario', help='task- or class- incremental learning')
parser.add_argument('--task-num', type = int, default = 19)
parser.add_argument('--in-size', type = int, default = 768)
parser.add_argument('--batch-size', type = int, default = 128, metavar = 'Batch', help='Batch size')
parser.add_argument('--class-per-task', type = int, default = 5, metavar = 'out', help='output size')
parser.add_argument('--task-sequence', type = int, nargs='+', default = [5,], metavar = 'Batch', help='Batch size')
parser.add_argument('--ep-inference', default = False, action = 'store_true', help='whether use the episode inference generated by ANN')

params = parser.parse_args()

from utils import create_dataset, SNN, aux_net
batch_size=128
wins=10
device=0
seed=222
params.m_th = [0]*5+[np.inf]*5
params.task_offset=[0]
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
if params.dataset == 'cifar100':
    params.task_num=19
    params.in_size=768
    params.class_per_task=5
    out_size=100
    epochs_per_task=5
    class_num=100
    hidden_neurons=64
    ann_channel=256
elif params.dataset == 'tinyimage':
    params.task_num=39
    params.in_size=768
    params.class_per_task=5
    out_size=200
    epochs_per_task=5
    class_num=200
    hidden_neurons=64
    ann_channel=256

if params.ep_inference:
    ann_path = './ANN_Prior/{0}_{1}/exp_gate{2}/'.format(params.scenario, params.dataset, hidden_neurons)
    snn_model_path = './snn_trained_model/{0}_{1}_EIF_CH_HNN/SNN_state_dict.pt'.format(params.dataset, params.scenario)
    ann_trained = aux_net(inputs_size=params.in_size, hidden_size_list=[hidden_neurons,hidden_neurons], 
                      channels=ann_channel, n_value=10).to(device)
    state_dict = torch.load(ann_path + '/models/gate_net.pth')
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'linear_2_1' in key:
            new_key = key.replace('linear_2_1', 'linear_2')
        else:
            new_key = key
        new_state_dict[new_key]=value
    ann_trained.load_state_dict(new_state_dict)
else:
    snn_model_path = './snn_trained_model/{0}_{1}_LIF_Baseline/SNN_state_dict.pt'.format(params.dataset, params.scenario)
    ann_trained = None

snn_model = SNN([params.in_size,hidden_neurons,hidden_neurons,out_size], norm='bn', track_BN_state=False).to(device)
snn_model.load_state_dict(torch.load(snn_model_path))
params.task_sequence+=[params.class_per_task for i in range(params.task_num)]

def generate_gate(data, ann, args):
    if params.scenario=='task-incre':
        meta_ann_10 = ann(data.reshape(data.size(0),-1).float(), scale=50)
        meta_ann = [np.argmax(meta_ann_10[i].mean(0).detach().cpu().numpy(),axis=1) for i in range(2)]
        gate = [torch.tensor(~np.in1d(meta_ann[i][0:hidden_neurons], np.where(np.isinf(args.m_th)))).to(device).reshape(hidden_neurons) for i in range(2)]
    elif params.scenario=='class-incre':
        meta_ann_10 = ann(data.reshape(data.size(0),-1).float(), scale=50)
        meta_ann = [np.argmax(meta_ann_10[i].detach().cpu().numpy(), axis=2) for i in range(2)]
        gate = [torch.tensor(~np.in1d(meta_ann[i][:,0:hidden_neurons], np.where(np.isinf(args.m_th)))).to(device).reshape(meta_ann[0].shape[0],hidden_neurons) for i in range(2)]

    return gate

def test(snn_model, test_loader, device, ann, verbose = True):
    
    snn_model.eval()
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        if params.ep_inference:
            gate = generate_gate(data, ann, params)
        else:
            gate=[]
        
        output = snn_model(data, gate)
        # test_loss += criterion(output, target).item() # mean batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_acc = round( 100. * float(correct) / (len(test_loader.dataset)), 2)
    
    if verbose :
        print('Test accuracy: {}/{} ({:.2f}%)'.format(
            correct, len(test_loader.dataset),
            test_acc))
    
    return test_acc

def main():
    _, test_loader_list,_ = create_dataset(params)
    acc_all=[]
    for task_idx, task in enumerate(test_loader_list):
        print('task_idx:', task_idx)
        test_accuracy = test(snn_model, task, device, ann_trained)
        acc_all.append(test_accuracy)
    acc_all=np.array(acc_all)
    # np.save('./Results/acc_all_sample.npy', acc_all)
    acc_5_class=np.average(acc_all.reshape(-1, 5), axis=1)
    print('Mean Accuracy:',acc_all, np.mean(acc_all), acc_5_class)

main()