import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
from optimizers_utils import SGD_meta, Adam_meta
import torch.nn as nn
from torch.utils.data.dataset import ConcatDataset
import copy
import spikingjelly.clock_driven.encoding as encoding
import tonic

probs = 0.0 
v_th_scales = 0.2
lens = 0.5
wins=10

class SignActivation(torch.autograd.Function): 

    @staticmethod
    def forward(ctx, i):
        result = i.sign()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_i = grad_output.clone()
        grad_i[i.abs() > 1.0] = 0
        return grad_i


class SG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        # print('input:',input.shape)
        out = (input > 0).float()
        # print('out:',out.shape)
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIF(torch.nn.Module):
    def __init__(self, v_th=1.0, tau=0.25, gamma=1.0):
        super(LIF, self).__init__()
        self.heaviside = SG.apply
        self.v_th = v_th
        self.tau = tau
        self.gamma = gamma

    def forward(self, x):#[N,T,C]
        mem_v = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)#decay
            # print('spike:',spike.shape)
            # print('spike:',spike.shape,'gate:',gate.shape)
            mem_v.append(spike)
            # mem_v.append(spike*gate)
            # print('gate:',gate)
        # print('re:',torch.stack(mem_v, dim=1).shape)
        # print('spike:',spike.shape, spike[0],'mem_v:', len(mem_v), mem_v[0].shape)
        return torch.stack(mem_v, dim=1)
    
class IF(nn.Module):
    def __init__(self, v_th=1.0):
        super(IF, self).__init__()
        self.heaviside = SG.apply
        self.v_th = v_th

    def forward(self, x):  # [N, T, C]
        mem_v = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem += x[:, t, ...]  # Integrate input
            spike = self.heaviside(mem - self.v_th, 1.0)  # Fire condition
            mem = mem * (1 - spike)  # Reset mechanism
            mem_v.append(spike)
        return torch.stack(mem_v, dim=1)
    
class Izhikevich(nn.Module):
    def __init__(self, a=0.02, b=0.25, c=-65, d=0.05):
        super(Izhikevich, self).__init__()
        self.a = nn.Parameter(torch.tensor([a], dtype=torch.float32), requires_grad=True)
        self.b = nn.Parameter(torch.tensor([b], dtype=torch.float32), requires_grad=True)
        self.c = nn.Parameter(torch.tensor([c], dtype=torch.float32), requires_grad=True)
        self.d = nn.Parameter(torch.tensor([d], dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        mem_v = []
        v = torch.full((x.shape[0], x.shape[2]), -65.0, device=x.device)  # Initial membrane potential
        u = v * self.b  # Recovery variable
        T = x.shape[1]
        for t in range(T):
            v += (0.04 * v ** 2 + 5 * v + 140 - u + x[:, t, ...])
            u += self.a * (self.b * v - u)
            spike = (v >= 30).float()
            v = torch.where(spike > 0, torch.full_like(v, self.c.item()), v)
            u = torch.where(spike > 0, u + self.d.item(), u)
            mem_v.append(spike)
        return torch.stack(mem_v, dim=1)
    
class EIF(nn.Module):
    def __init__(self, v_th=1.0, delta_T=0.2, tau=1, v_reset=0):
        super(EIF, self).__init__()
        self.v_th = v_th        # Threshold voltage
        self.delta_T = delta_T  # Sharpness of the exponential term
        self.tau = tau          # Time constant
        self.v_reset = v_reset  # Reset voltage

    def forward(self, x):  # Input shape [N, T, C]
        # Ensure the input x is a float tensor
        x = x.float()
        
        # Initialize membrane potential with float type
        mem = torch.full((x.shape[0], x.shape[2]), self.v_reset, dtype=torch.float, device=x.device)
        
        mem_v = []  # To collect spikes
        T = x.shape[1]  # Number of timesteps
        for t in range(T):
            # Update membrane potential
            mem += (x[:, t, ...] - mem + self.delta_T * torch.exp((mem - self.v_th) / self.delta_T)) / self.tau
            # Spike condition
            spike = (mem >= self.v_th).float()
            # Reset membrane potential where spikes occur
            mem = torch.where(spike > 0, torch.full_like(mem, self.v_reset), mem)
            mem_v.append(spike)
        
        return torch.stack(mem_v, dim=1)

class SeqToANNContainer(torch.nn.Module):
    # Altered form SpikingJelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = torch.nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        # print('x_seq:',x_seq) #[B,10,784]
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        # print('y_shape:',y_shape)[B,10]
        # print('x_seq:',x_seq[0])
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())#[1280,64]
        # print('y_seq:',y_seq)
        y_shape.extend(y_seq.shape[1:])#[1280,64]
        # print('y_seq_view:',y_seq.view(y_shape).shape) #[B,10,64]
        return y_seq.view(y_shape)#[B,10,64]
    
class SNN(nn.Module):
    def __init__(self,layers_dims, wins=10, tau=0.25, norm='bn', norm_arc='all', neu_model='LIF', track_BN_state=False, out_off=0):
        super(SNN, self).__init__()
        self.ewc_lambda = 5000  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = False # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = 5  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0
        self.layers_dims=layers_dims
        self.hidden_layers=len(self.layers_dims)-2
        self.encoder = encoding.PoissonEncoder()
        self.tau=tau
        self.norm=norm
        self.wins=wins
        self.norm_arc=norm_arc
        self.neu_model = neu_model
        self.out_off = out_off
        self.track_BN_state=track_BN_state
        layer_list=[]
        for layer in range(self.hidden_layers): 
            layer_list = layer_list + [( ('fc'+str(layer+1)) , SeqToANNContainer(
            nn.Linear(self.layers_dims[layer], self.layers_dims[layer+1]))) ]
            if self.norm=='bn':
                # layer_list = layer_list + [( ('bn'+str(layer+1)), TEBN(self.layers_dims[layer+1],wins,device)) ]
                layer_list = layer_list + [( ('bn'+str(layer+1)), nn.BatchNorm1d(self.layers_dims[layer+1], affine = True, track_running_stats = self.track_BN_state))]
            if self.neu_model=='LIF':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , LIF(tau=self.tau)) ]
            elif self.neu_model=='IF':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , IF()) ]
            elif self.neu_model == 'EIF':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , EIF(tau=self.tau)) ]
            elif self.neu_model == 'IKV':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , Izhikevich(d=self.tau)) ]

        layer_list = layer_list + [( 'fc3' , SeqToANNContainer(
            nn.Linear(self.layers_dims[self.hidden_layers], self.layers_dims[self.hidden_layers+1])))]
        self.layers = nn.ModuleDict(OrderedDict( layer_list ))
        self.heads = torch.nn.ModuleList()

    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(SeqToANNContainer(
            torch.nn.Linear(self.layers_dims[self.hidden_layers], num_outputs+self.out_off)))

    def forward(self, input, gate, return_features=False):
        if len(input.shape)==5:#dvs dataset
            input_ = input[:,:self.wins,...]
        elif len(input.shape)==4:#MNIST
            input_ = input.repeat(1,self.wins,1,1)
            for batchi in range(input.shape[0]):
                for step in range(self.wins):
                    encoded_input=self.encoder(input[batchi])
                    input_[batchi,step,:,:]=encoded_input
        else:#tinyimage or cifar100
            input=torch.unsqueeze(input, dim=1)
            input_=input.repeat(1,self.wins,1)
            for batchi in range(input.shape[0]):
                for step in range(self.wins):
                    encoded_input=self.encoder(input[batchi])
                    # print('encoded_input:',encoded_input.shape)#[1,28,28]
                    input_[batchi,step,:]=encoded_input
        x=input_.view(-1,self.wins,self.layers_dims[0]).float() #[B,10,784]
        for layer in range(self.hidden_layers):
            if True:#len(gate)==0 or layer==2:
                x = self.layers['lf'+str(layer+1)](x)
                # print('x:',x.shape)
            if len(gate)==0 or layer==2:
                # print('x:',x.shape)
                x = self.layers['fc'+str(layer+1)](x)
            else:
                tmp=gate[layer]
                if tmp.shape[0]>1:
                    x = self.layers['fc'+str(layer+1)](x).transpose(0,1)
                    x = x*tmp
                    x = x.transpose(0,1)
                else:
                    x = self.layers['fc'+str(layer+1)](x)*tmp
            if self.norm=='bn' and (layer==1 or self.norm_arc=='all'):
                x = self.layers['bn'+str(layer+1)](x.view(-1, self.layers_dims[layer+1]))
                x = x.view(-1,self.wins,self.layers_dims[layer+1])

        if len(self.heads) > 0:
            y = []
            for head in self.heads:
                outs=head(x)
                y.append(outs.mean(1))#normalize the FR among the time windows
            return y
        else:
            x = self.layers['fc3'](x)
            return x.mean(1)
        
    def save_bn_states(self):
        bn_states = []
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers):
                bn = copy.deepcopy(self.layers['bn'+str(l+1)].state_dict())
                bn_states.append(bn)
        if 'ln1' in self.layers.keys():
            for l in range(self.hidden_layers):
                bn = copy.deepcopy(self.layers['ln'+str(l+1)].state_dict())
                bn_states.append(bn)
        return bn_states
    
    def load_bn_states(self, bn_states):
        if 'bn1' in self.layers.keys():  
            for l in range(self.hidden_layers):
                self.layers['bn'+str(l+1)].load_state_dict(bn_states[l])
        if 'ln1' in self.layers.keys():
            for l in range(self.hidden_layers):
                self.layers['ln'+str(l+1)].load_state_dict(bn_states[l])
    
    def reset_bn_states(self,):
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers):
                self.layers['bn'+str(l+1)].reset_running_stats()

class SNN_pMNIST(torch.nn.Module):
    def __init__(self, layers_dims, wins=10,tau=0.25,norm='bn',neu_model='LIF',tracking_stat=False):
        super(SNN_pMNIST, self).__init__()
        self.ewc_lambda = 5000  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = False # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = 5  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0
        self.layers_dims=layers_dims
        self.hidden_layers=len(self.layers_dims)-2
        self.encoder = encoding.PoissonEncoder()
        self.tau=tau
        self.norm=norm
        self.neu_model = neu_model
        self.wins=wins

        layer_list=[]
        for layer in range(self.hidden_layers): 
            layer_list = layer_list + [( ('fc'+str(layer+1)) , SeqToANNContainer(
            torch.nn.Linear(self.layers_dims[layer], self.layers_dims[layer+1]))) ]
            if self.norm=='bn':
                # layer_list = layer_list + [( ('bn'+str(layer+1)), TEBN(self.layers_dims[layer+1],wins,device)) ]
                layer_list = layer_list + [( ('bn'+str(layer+1)), torch.nn.BatchNorm1d(self.layers_dims[layer+1], affine = True, track_running_stats = tracking_stat))]
            if self.neu_model=='LIF':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , LIF(tau=self.tau)) ]
            elif self.neu_model=='IF':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , IF()) ]
            elif self.neu_model == 'EIF':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , EIF(tau=self.tau)) ]
            elif self.neu_model == 'IKV':
                layer_list = layer_list + [(  ('lf'+str(layer+1) ) , Izhikevich(d=self.tau)) ]
        layer_list = layer_list + [( 'fc3' , SeqToANNContainer(
            torch.nn.Linear(self.layers_dims[self.hidden_layers], self.layers_dims[self.hidden_layers+1])))]
        self.layers = torch.nn.ModuleDict(OrderedDict( layer_list ))

    def forward(self, input, gate):
        input_=input.repeat(1,self.wins,1,1)
        for batchi in range(input.shape[0]):
            for step in range(self.wins):
                encoded_input=self.encoder(input[batchi])
                input_[batchi,step,:,:]=encoded_input
        x=input_.view(-1,self.wins,self.layers_dims[0]) #[B,10,784]

        for layer in range(self.hidden_layers):
            x = self.layers['lf'+str(layer+1)](x)
            if len(gate)==0 or layer==2:
                x = self.layers['fc'+str(layer+1)](x)
            else:
                tmp=gate[layer]
                if tmp.shape[0]>1:
                    x = self.layers['fc'+str(layer+1)](x).transpose(0,1)
                    x = x*tmp
                    x = x.transpose(0,1)
                else:
                    x = self.layers['fc'+str(layer+1)](x)*tmp
            if self.norm=='bn':
                x = self.layers['bn'+str(layer+1)](x.view(-1, self.layers_dims[layer+1]))
                x = x.view(-1,self.wins,self.layers_dims[layer+1])

        x = self.layers['fc3'](x)
        outs = x.mean(1)
        return outs
    
    def save_bn_states(self):
        bn_states = []
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers):
                bn = copy.deepcopy(self.layers['bn'+str(l+1)].state_dict())
                bn_states.append(bn)
        return bn_states
    
    def load_bn_states(self, bn_states):
        if 'bn1' in self.layers.keys():  
            for l in range(self.hidden_layers):
                self.layers['bn'+str(l+1)].load_state_dict(bn_states[l])
    
    def reset_bn_states(self,):
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers):
                self.layers['bn'+str(l+1)].reset_running_stats()

class aux_net(nn.Module):
    def __init__(self, inputs_size, hidden_size_list, channels, n_value):
        super(aux_net, self).__init__()
        self.n_value = n_value
        self.linear_1 = nn.Linear(inputs_size, channels)
        self.linear_2 = nn.Linear(channels, channels)
        # self.linear_3 = nn.Linear(channels, channels)
        self.heads = []
        for i in range(len(hidden_size_list)):
            self.heads.append(
                nn.Linear(channels, hidden_size_list[i]*n_value))
        self.heads = nn.ModuleList(self.heads)

    def forward(self, x, scale):

        temp_1 = self.linear_1(x)
        temp_1 = torch.relu(temp_1)

        temp_2 = self.linear_2(temp_1)
        temp_2 = torch.relu(temp_2)

        # temp_3 = self.linear_3(temp_2)
        # temp_3 = torch.relu(temp_3)

        temp_3 = temp_2

        heads = []
        for head in self.heads:
            buf = head(temp_3)
            buf = buf.reshape(buf.size(0), -1, self.n_value)
            buf = torch.softmax(buf*scale, dim=2)
            heads.append(buf)
        return heads
    
class ANN(torch.nn.Module):
    def __init__(self,layers_dims,norm='bn',track_BN_state=False):
        super(ANN, self).__init__()
        self.ewc_lambda = 5000  # -> hyperparam: how strong to weigh EWC-loss ("regularisation strength")
        self.gamma = 1.  # -> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = False # -> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = 5  # -> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = False  # -> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0
        self.layers_dims=layers_dims
        self.hidden_layers=len(self.layers_dims)-2
        self.norm=norm
        self.track_BN_state=track_BN_state
        # self.out_off=out_off
        layer_list=[]
        for layer in range(self.hidden_layers):
            layer_list = layer_list + [( ('fc'+str(layer+1)) , nn.Linear(self.layers_dims[layer], self.layers_dims[layer+1]))]
            if self.norm=='bn':
                layer_list = layer_list + [( ('bn'+str(layer+1)), nn.BatchNorm1d(self.layers_dims[layer+1], affine = True, track_running_stats = self.track_BN_state))]
        # print('outlayer:', self.layers_dims[self.hidden_layers+1])
        layer_list = layer_list + [( 'fc3' , nn.Linear(self.layers_dims[self.hidden_layers], self.layers_dims[self.hidden_layers+1])) ]

        self.layers = torch.nn.ModuleDict(OrderedDict( layer_list ))
        self.heads = torch.nn.ModuleList()


    def add_head(self, num_outputs):
        """Add a new head with the corresponding number of outputs. Also update the number of classes per task and the
        corresponding offsets
        """
        self.heads.append(
            nn.Linear(self.layers_dims[self.hidden_layers], num_outputs))

    def forward(self, input, gate, return_features=False):
        if len(input.shape)==5:#dvs dataset
            input = torch.mean(input.float(), dim=1)
        x=input.view(-1,self.layers_dims[0]).float()
        for layer in range(self.hidden_layers):
            if len(gate)==0 or layer==2:
                x = self.layers['fc'+str(layer+1)](x)
            else:
                tmp=gate[layer]+0
                x = self.layers['fc'+str(layer+1)](x)
                x=x*tmp
            if self.norm=='bn':
                x = self.layers['bn'+str(layer+1)](x)
    
        if len(self.heads) > 0:
            y = []
            for head in self.heads:
                outs=head(x)
                y.append(outs.mean(1))#normalize the FR among the time windows
            return y
        else:
            x = self.layers['fc3'](x)
            return x
    
    def save_bn_states(self):
        bn_states = []
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers):
                bn = copy.deepcopy(self.layers['bn'+str(l+1)].state_dict())
                bn_states.append(bn)
        return bn_states
    
    def load_bn_states(self, bn_states):
        if 'bn1' in self.layers.keys():  
            for l in range(self.hidden_layers):
                self.layers['bn'+str(l+1)].load_state_dict(bn_states[l])
    
    def reset_bn_states(self,):
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers):
                self.layers['bn'+str(l+1)].reset_running_stats()


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]
        
    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return data, label

    def __len__(self):
        return self.length

def generate_gate(data, target, ann, args):
    sorted_target, indices = target.sort(dim=0)
    _,return_indices=indices.sort(dim=0)
    # print('indices:', indices)
    sorted_data = data[indices,:]
    data_gen = sorted_data.reshape(sorted_data.size(0),-1).float()
    meta_ann_10 = ann(data_gen, scale=50)
    target_label=torch.unique(sorted_target)
    # print('target_label:', target_label)
    temp=[]
    temp_ = []
    for i, labels in enumerate(target_label):
        index_ = torch.where(sorted_target==labels)
        meta_ann = [np.argmax(meta_ann_10[j][index_].mean(0).detach().cpu().numpy(), axis=1) for j in range(2)]
        # print('index_:', len(index_[0]),'meta_ann:', meta_ann[0].shape)
        temp.append(torch.tensor(~np.in1d(meta_ann[0][0:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).cuda().repeat(len(index_[0]),1))
        temp_.append(torch.tensor(~np.in1d(meta_ann[1][0:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).cuda().repeat(len(index_[0]),1))
    gate=[torch.cat(temp,dim=0)[return_indices], torch.cat(temp_,dim=0)[return_indices]]

    return gate

def gate_noise(data, ann, args):

    data_gen = data.reshape(data.size(0),-1).float()
    meta_ann_10 = ann(data_gen, scale=50)
    if args.N_noise>1:
        noise_data_0=torch.zeros(args.N_noise+1, *meta_ann_10[0].shape).cuda()
        noise_data_1=torch.zeros(args.N_noise+1, *meta_ann_10[1].shape).cuda()
        for i in range(args.N_noise):
            noisy_data = data_gen + torch.randn_like(data_gen)*args.noise_factor
            tmp_0, tmp_1 = ann(noisy_data, scale=50)
            noise_data_0[i] = tmp_0
            noise_data_1[i] = tmp_1
        noise_data_0[-1] = meta_ann_10[0]
        noise_data_1[-1] = meta_ann_10[1]
        # print(noise_data_0.shape)
        meta_ann = [torch.argmax(noise_data_0.mean(0),dim=2).cpu().numpy(), torch.argmax(noise_data_1.mean(0),dim=2).cpu().numpy()]
        # print(meta_ann[0].shape)
    else:
        noise_data = data_gen + torch.randn_like(data_gen)*args.noise_factor
        meta_ann_10_new = ann(noise_data, scale=50)
        meta_ann = [(meta_ann_10[i]+meta_ann_10_new[i]).div(2).argmax(dim=2).cpu().numpy() for i in range(2)] 
    gate = [torch.tensor(~np.in1d(meta_ann[i][:,:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).cuda().view(-1,args.hidden_layers[0]) for i in range(2)]

    return gate

def train(model, ann, train_loader, current_task_index, optimizer, device, args, gate,
          prev_cons=None, prev_params=None, path_integ=None):
    
    reg_loss=0
    model.train()
    # for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        if args.dataset=='dvsG':
            indices = torch.randint(0, data.size(1), (args.wins,))
            data = data[:,indices,...]
        optimizer.zero_grad()
        if args.gate and args.gate_type=='generate-train-test':
            # print('data:', data.shape)
            if args.dataset=='dvsG':
                data_ann = torch.mean(data, dim=1)
                data_gen = data_ann.reshape(data_ann.size(0),-1).float()
            else:
                data_gen = data.reshape(data.size(0),-1).float()
            meta_ann_10 = ann(data_gen, scale=50)
            if args.sort_gate_train=="mean":
                meta_ann = [torch.argmax(meta_ann_10[i].mean(0), dim=1).cpu().numpy() for i in range(2)]
                gate = [torch.tensor(~np.in1d(meta_ann[i][:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).to(device).reshape(args.hidden_layers[0]) for i in range(2)]
            else:
                meta_ann = [torch.argmax(meta_ann_10[i], dim=2).cpu().numpy() for i in range(2)]
                gate = [torch.tensor(~np.in1d(meta_ann[i][:,:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).to(device).view(-1,args.hidden_layers[0]) for i in range(2)]

        if args.scenario=='class-incre' and args.type_of_head!='mask':# or args.type_of_head=='active'):
            output = model(data, gate)
            loss = criterion(output[current_task_index], target, args.task_offset[current_task_index])
        elif args.scenario=='class-incre' and args.type_of_head=='mask':
            output = model(data, gate)
            mask = torch.ones_like(output)*(-np.inf)
            mask = mask.to(device)
            mask[:,current_task_index*args.class_per_task:(current_task_index+1)*args.class_per_task] = 0
            output = output+mask
            loss = criterion(output, target)
        elif args.scenario=='task-incre':
            output = model(data, gate)
            loss = criterion(output, target, args.task_offset[current_task_index])
        elif args.scenario=='upperbound-class' or args.scenario=='upperbound-task':
            output = model(data, gate=[])
            loss = criterion(output, target)

        if args.ewc:
            ewc_loss = EWC_loss(model, prev_cons, prev_params, current_task_index, device, ewc_lambda=args.ewc_lambda)
            total_loss = loss + ewc_loss
        elif args.si:
            p_prev, p_old = prev_params
            si_loss = SI_loss(model, prev_cons, p_prev, args.si_lambda)
            total_loss = loss + si_loss
        else:
            total_loss = loss

        total_loss.backward()
        optimizer.step()
        
        if args.si:
            update_W(model, path_integ, p_old, args)

    
def test(model, ann, test_loader, current_task_index, device, args, gate, verbose = False):
    
    model.eval()
    test_loss, total_acc_tag, batch_id = 0, 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_id+=1
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device)
            if args.gate and args.gate_type=='generate-train-test':
                if args.dataset=='dvsG':
                    data_ann = torch.mean(data, dim=1)
                    data_gen = data_ann.reshape(data_ann.size(0),-1).float()
                else:
                    data_gen = data.reshape(data.size(0),-1).float()
                meta_ann_10 = ann(data_gen, scale=50)
                if args.sort_gate_test=='mean':
                    meta_ann = [np.argmax(meta_ann_10[i].mean(0).detach().cpu().numpy(),axis=1) for i in range(2)]
                    gate = [torch.tensor(~np.in1d(meta_ann[i][0:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).to(device).reshape(args.hidden_layers[0]) for i in range(2)]
                else:
                    meta_ann = [np.argmax(meta_ann_10[i].detach().cpu().numpy(),axis=2) for i in range(2)]
                    gate = [torch.tensor(~np.in1d(meta_ann[i][:,0:args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).to(device).reshape(meta_ann[0].shape[0],args.hidden_layers[0]) for i in range(2)]
            
            if args.scenario=='class-incre' and args.type_of_head=='mask':
                output = model(data, gate)
                loss = criterion(output, target)
            elif args.scenario=='class-incre' and args.type_of_head!='mask':
                output = model(data, gate)
                loss = criterion(output[current_task_index], target, args.task_offset[current_task_index])
            elif args.scenario=='task-incre':
                output = model(data, gate)
                loss = criterion(output, target, args.task_offset[current_task_index])
            elif args.scenario=='upperbound-class' or args.scenario=='upperbound-task':
                output = model(data, gate=[])
                loss = criterion(output, target)

            test_loss += loss.item() # mean batch loss
            _, hits_tag = calculate_metrics(output, target, current_task_index, args)
            total_acc_tag += hits_tag.sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc_tag = round( 100. * total_acc_tag / len(test_loader.dataset), 2)
    if verbose :        
        print('Test accuracy Tag: ({:.2f}%)'.format(
            test_acc_tag))
    
    return test_acc_tag, test_loss

def calculate_metrics(outputs, targets, this_task, args):
    """Contains the main Task-Aware and Task-Agnostic metrics"""
    if args.scenario=='class-incre' and args.type_of_head=='head':
        pred_taw = torch.zeros_like(targets)
        # Task-Aware Multi-Head
        for m in range(len(pred_taw)):#BatchSize
            # print('this_task:', this_task, 'outputs:', len(outputs), outputs[0].shape, args.task_offset, 'task_idx:', this_task)
            pred_taw[m] = outputs[this_task][m].argmax()
            # print('task_offset_in_eval:', self.model.task_offset[this_task])#0/25/50/75
        hits_taw = (pred_taw == targets).float()
        pred_tag = torch.cat(outputs, dim=1)
        pred_tag = pred_tag.argmax(1)
        hits_tag = (pred_tag == targets).float()
    elif args.scenario=='class-incre' and args.type_of_head=='mask':
        pred_tag = outputs.argmax(1)
        # print('pred_tag:',pred_tag)
        hits_tag = (pred_tag == targets).float()
        hits_taw = hits_tag
    elif args.scenario=='task-incre':
        pred_taw = outputs.argmax(1)
        hits_taw = (pred_taw == (targets-args.task_offset[this_task])).float()
        hits_tag = hits_taw
    elif args.scenario=='upperbound-class' or args.scenario=='upperbound-task':
        pred_taw = outputs.argmax(1)
        hits_taw = (pred_taw == targets).float()
        hits_tag = hits_taw
    return hits_taw, hits_tag

def create_permuted_loaders(permut, batchsize):
    
    # permut = torch.from_numpy(np.random.permutation(784))
        
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                      torchvision.transforms.Lambda(lambda x: x.view(-1)[permut].view(1, 28, 28) ),
                      torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])

    dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batchsize, shuffle=True, num_workers=1)

    dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batchsize, shuffle=False, num_workers=1)


    return train_loader, test_loader, dset_train

def trans_indices(dataset_, class_list):
    indices = []
    for idx in range(len(dataset_)):
        label = dataset_.targets[idx]  # Access target directly if possible
        if label in class_list:
            indices.append(idx)
    return indices
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    # Separate data and labels (if labels are part of your batch)
    data, labels = zip(*batch)
    
    # Convert data from numpy arrays to tensors
    data = [torch.tensor(d, dtype=torch.float32) for d in data]
    
    # Pad the sequences
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    
    # Convert labels to a tensor
    labels = torch.tensor(labels)
    
    return data_padded, labels

def create_dataset(args):
    print('Creating dataset...')
    train_loader_list=[]
    test_loader_list=[]
    dset_train_list=[]
    task_num=0
    if args.dataset=='pMNIST':
        permutation_list=np.load(args.ms_path+'permutation_index_for_training_SNN.npy')
        for idx in range(len(args.task_sequence)):
            # print('task:',task[1:])
            # print(len(args.task_sequence),permutation_list.shape)
            train_loader, test_loader, dset_train = create_permuted_loaders(permutation_list[idx,:], batchsize=args.batch_size)#'MNIST'
            # print('train_data:',len(train_loader),'test_data:',len(test_loader))
            train_loader_list.append(train_loader)
            test_loader_list.append(test_loader)
            dset_train_list.append(dset_train)
            args.task_offset+=[task_num]
    elif args.dataset=='sMNIST' or args.dataset=='dvsG':
        if args.dataset=='sMNIST':
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))])
            dset_train = torchvision.datasets.MNIST('./mnist_pytorch', train=True, transform=transform, target_transform=None, download=True)
            dset_test = torchvision.datasets.MNIST('./mnist_pytorch', train=False, transform=transform, target_transform=None, download=True)
            for i in range(len(args.task_sequence)):
                class_id = np.array(np.arange(task_num, args.task_sequence[i]+task_num))
                subset_trainset = torch.utils.data.Subset(dset_train, [i for i in range(len(dset_train)) if dset_train[i][1] in class_id])
                subset_testset = torch.utils.data.Subset(dset_test, [i for i in range(len(dset_test)) if dset_test[i][1] in class_id])
                trainloader = torch.utils.data.DataLoader(subset_trainset, batch_size=args.batch_size, shuffle=True)
                testloader = torch.utils.data.DataLoader(subset_testset, batch_size=args.batch_size, shuffle=False)
                task_num+=args.task_sequence[i]
                args.task_offset+=[task_num]
                train_loader_list.append(trainloader)
                test_loader_list.append(testloader)
                dset_train_list.append(subset_trainset)
        elif args.dataset=='dvsG':
            frame_transform = tonic.transforms.ToFrame(sensor_size=tonic.datasets.DVSGesture.sensor_size, time_window=100000)
            # Chain the transforms
            dset_train = tonic.datasets.DVSGesture(save_to='/data1/sqq/dataset/dvsG', transform=frame_transform, train=True)
            dset_test = tonic.datasets.DVSGesture(save_to='/data1/sqq/dataset/dvsG', transform=frame_transform, train=False)
            for i in range(len(args.task_sequence)):
                class_id = np.arange(task_num, args.task_sequence[i]+task_num)
                indices_train = trans_indices(dset_train, class_id)
                indices_test = trans_indices(dset_test, class_id)
                subset_trainset = torch.utils.data.Subset(dset_train, indices_train)
                subset_testset = torch.utils.data.Subset(dset_test, indices_test)
                trainloader = torch.utils.data.DataLoader(subset_trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn)
                testloader = torch.utils.data.DataLoader(subset_testset, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=collate_fn)
                task_num+=args.task_sequence[i]
                args.task_offset+=[task_num]
                train_loader_list.append(trainloader)
                test_loader_list.append(testloader)
                dset_train_list.append(subset_trainset)

    elif args.dataset=='cifar100' or args.dataset=='tinyimage':
        path = 'embeddings/{}_clip_vitl14'.format(args.dataset)
        train_inputs=np.load(path+'/train/inputs.npy')
        train_target=np.load(path+'/train/labels.npy')
        test_inputs=np.load(path+'/test/inputs.npy')
        test_target=np.load(path+'/test/labels.npy')

        for i in range(len(args.task_sequence)):
            class_id = np.arange(task_num, args.task_sequence[i]+task_num)
            class_label = [np.where(np.isin(train_target, class_id)), np.where(np.isin(test_target, class_id))]
            train_label=train_target[class_label[0]]
            test_label=test_target[class_label[1]]
            if args.scenario=='upperbound-task':
                train_data = MyDataSet(data=train_inputs[class_label[0]], label=train_label-task_num)
                test_data = MyDataSet(data=test_inputs[class_label[1]], label=test_label-task_num)
                test_loader =  torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
                test_loader_list.append(test_loader)
            else:
                train_data = MyDataSet(data=train_inputs[class_label[0]], label=train_label)
                test_data = MyDataSet(data=test_inputs[class_label[1]], label=test_label)
                test_loader =  torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
                test_loader_list.append(test_loader)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
            train_loader_list.append(train_loader)
            dset_train_list.append(train_data)
            task_num+=args.task_sequence[i]
            args.task_offset+=[task_num]
    if args.scenario=='upperbound-class' or args.scenario=='upperbound-task':
        combined_train_dataset=ConcatDataset(dset_train_list)
        combined_train_loader=torch.utils.data.DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True)  
        return [combined_train_loader], test_loader_list, dset_train_list
    else:
        return train_loader_list, test_loader_list, dset_train_list

def createHyperparametersFile(path, args):

    hyperparameters = open(path + r"/hyperparameters.txt","w+")
    L = [
        "- hidden layers: {}".format(args.hidden_layers) + "\n",
        "- normalization: {}".format(args.norm) + " ,tracking_BN_state={}".format(args.track_BN_stat) + "\n",
        "- net: {}".format(args.net) + "\n",
        "- lr: {}".format(args.lr) + "\n",
        "- gamma: {}".format(args.gamma) + "\n",
        "- decay: {}".format(args.decay) + "\n",
        "- epochs per task: {}".format(args.epochs_per_task) + "\n",
        "- batch size: {}".format(args.batch_size) + "\n",
        "- seed: {}".format(args.seed) + "\n",
        "- tau_snn: {}".format(args.tau) + "\n",
        "- meta value: {}".format(args.meta) + "\n",
        "- meta activation: {}".format(args.meta_ac) + "\n",
        "- scenario: {}".format(args.scenario) + "\n",
        "- task number: {}".format(args.task_num) + "\n", 
        "- task sequence: {}".format(args.task_sequence) +"\n", 
        "- gate or not: {}".format(args.gate) + "\n",
        "- xdg or not: {}".format(args.xdg) +"\n",
        "- meta: {}".format(args.meta) + "\n",
        "- ann-meta: {}".format(args.ann_meta) + "\n",
        "- meta one-hot value: {}".format(args.m_th) + "\n",
        "- optimizer: {}".format(args.optim) + "\n",
        "- ewc :{}".format(args.ewc),
        "- ewc lambda: {}".format(args.ewc_lambda) + "\n",
        "- si :{}".format(args.si),
        "- si lambda: {}".format(args.si_lambda) + "\n",
        "- head type: {}".format(args.type_of_head) + "\n",
        "- reset BN: {}".format(args.reset_BN) + "\n",
        "- out size: {}".format(args.out_size) + "\n",
        "- Loss function: {}".format(args.loss) + "\n",
        "- Out OFF: {}".format(args.out_off) + "\n",
        "- gate probability in xdg: {}".format(args.gate_prob) + "\n",
        "- gate generated by ANN in {}".format(args.gate_type) + "\n",
        "- meta path: {}".format(args.ms_path) + "\n",
        "- load bn: {}".format(args.load_bn)+"\n",
        "- sort gate in training: {}".format(args.sort_gate_train) + "\n",
        "- sort gate in testing: {}".format(args.sort_gate_test) + "\n",
        "- dataset: {}".format(args.dataset) + "\n",
        "- ANN channel: {}".format(args.ann_channels) + "\n",
        "- time window length: {}".format(args.wins) + "\n",
        "- neuron model: {}".format(args.neuron_model) + " -tau: {}".format(args.tau) + "\n",]
    
    hyperparameters.writelines(L)
    hyperparameters.close()

def save_parameters(model, path, epoch, save=True):
    
    weights_all=[]
    for (n, p) in model.named_parameters():
        # print('bias?:',n,n.find('bias'))
        if (n.find('bias') == -1) and (n.find('bn') == -1) and (len(p.size()) != 1):  #bias or batchnorm weight -> no plot, only plot the fc weight layer

            weights_all.append(p.data.cpu().numpy())            #TVGG or FVGG plot p

    if save:
        time = datetime.now().strftime('%H-%M-%S')
        np.save(path+'/'+time+'_'+str(epoch)+'_weight.npy',weights_all)

def plot_parameters(model, path, epoch, plot_num, save=True):
    
    fig = plt.figure(figsize=(15, 10))
    i = 1

    for (n, p) in model.named_parameters():
        # print('bias?:',n,n.find('bias'))
        if (n.find('bias') == -1) and (n.find('bn') == -1) and (len(p.size()) != 1):# and (p.size()[0]!=100):  #bias or batchnorm weight -> no plot, only plot the fc weight layer
            fig.add_subplot(2,2,i)
            if model.__class__.__name__.find('B') != -1:  #BVGG -> plot p.org
                if hasattr(p,'org'):
                    weights = p.org.data.cpu().numpy()#"p.org" is non-binary orginal weights
                else:
                    weights = p.data.cpu().numpy()
                binet = 100 #for plot
            else:
                weights = p.data.cpu().numpy()            #TVGG or FVGG plot p
                binet = 50
            i+=1
            plt.title( n.replace('.','_') ,fontsize=20)
            plt.hist( weights.flatten(), binet)

    if save:
        time = datetime.now().strftime('%H-%M-%S')
        fig.savefig(path+'/'+time+'_'+str(epoch)+'_weight_distribution.png')
    plt.close()
    
def plot_last_weight(model, path, task_id, epoch, args):
    
    fig = plt.figure(figsize=(15, 10))
    out_layer = 100//args.class_per_task
    for i in range(0,out_layer):
        fig.add_subplot(10,out_layer//10,i+1)
        if args.net=='ann':
            weights=model.layers.fc3.weight[i*args.class_per_task].data.cpu().numpy()
        elif args.net=='snn':
            weights=model.layers.fc3.module.weight[i*args.class_per_task].data.cpu().numpy()
        plt.title( '{0}task_{1}epoch'.format(i, epoch),fontsize=5)
        plt.hist( weights.flatten())
        time = datetime.now().strftime('%H-%M-%S')
        fig.savefig(path+'/'+'{0}task_{1}epoch_weight_distribution.png'.format(task_id, epoch))
    plt.close()

def plot_results(data,path,length,args,save=True):
    
    y_taw, y_tag=[], []
    for idx in range(length):
        if args.scenario=='upperbound':
            y_tag.append([data['acc_test_tsk_'+str(idx+1)][len(data)-1]])
        else:
            # y_taw.append([data['acc_test_taw_tsk_'+str(idx+1)][args.epochs_per_task*len(args.task_sequence)-1]])
            y_tag.append([data['acc_test_tsk_'+str(idx+1)][len(data)-1]])
    x=range(1,len(y_tag)+1)
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 20,
            }
    fig = plt.figure(figsize=(15, 10))
    plt.plot(x,y_tag,color='red',marker='o',linewidth=6)
    plt.axis([0,len(y_tag)+1,0,100])
    plt.title('Performance',fontdict=font)
    plt.xlabel('Task',fontdict=font) #, backgroundcolor='grey')
    plt.ylabel('Test accuracy(%)',fontdict=font) #, backgroundcolor='grey')
    if save:
        time = datetime.now().strftime('%H-%M-%S')
        fig.savefig(path+'/'+time+'performance_results.png')

    plt.close()

def meta_detect(x):
    x[torch.isnan(x)]=0
    return x

def criterion(outputs, targets, t=0):
    """Returns the loss value"""
    return torch.nn.functional.cross_entropy(outputs, targets - t)

def meta_seg_onehot(x,m_th):
    y=[]
    for i in range(len(x)):
        y.append(m_th[int(x[i])])

    return y

def estimate_fisher(model, dataset, args, device, current_task_idx, num = 1000, empirical = True):
    # Estimate the FI-matrix for num batches of size 1
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    est_fisher_info = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            n = n.replace('.', '__')
            est_fisher_info[n] = p.detach().clone().zero_()
    
    model.eval()
    for index,(x,y) in enumerate(loader):
        # break from for-loop if max number of samples has been reached

        if index >= num:
            break
        # run forward pass of model
        x = x.to(device)
        output = model(x, gate=[])
        if empirical:
            # -use provided label to calculate loglikelihood --> "empirical Fisher":
            label = torch.LongTensor([y]) if type(y)==int else y
            label = label.to(device)
        else:
            # -use predicted label to calculate loglikelihood:
            label = output.max(1)[1]
        # calculate negative log-likelihood
        if args.scenario=='task-incre':
            negloglikelihood = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), label-args.task_offset[current_task_idx])
        else:
            negloglikelihood = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(output, dim=1), label)

        # Calculate gradient of negative loglikelihood
        model.zero_grad()
        negloglikelihood.backward()

        # Square gradients and keep running sum
        for n, p in model.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                if p.grad is not None:
                    est_fisher_info[n] += p.grad.detach() ** 2 #first-order derivative

    est_fisher_info = {n: p/index for n, p in est_fisher_info.items()}
    
    return est_fisher_info



def EWC_loss(model, previous_tasks_fisher, previous_tasks_parameters, current_task_index, device, ewc_lambda=5000):
    
    if current_task_index == 0: #no task to remember -> return 0
        return torch.tensor(0.).to(device)
    else:
        losses = []
        for task_idx in range(current_task_index): # for all previous tasks and parameters
            for n, p in model.named_parameters():
                if ((p.requires_grad) and (n.find('bn') == -1)):
                    n = n.replace('.', '__')
                    mean = previous_tasks_parameters[n][task_idx]
                    fisher = previous_tasks_fisher[n][task_idx]
                    #print('in ewc loss, param =', p[0,0])
                    losses.append((fisher * (p-mean)**2).sum())
        return ewc_lambda*(1./2)*sum(losses)


def update_omega(model, omega, p_prev, W, epsilon=0.1):
    for n, p in model.named_parameters():
        if n.find('bn') == -1: # not batchnorm
            if p.requires_grad:
                n = n.replace('.', '__')
                # if isinstance(model, BNN):
                #     p_current = p.org.detach().clone()   # sign()
                # else:
                p_current = p.detach().clone()
                p_change = p_current - p_prev[n]
                omega_add = W[n]/(p_change**2 + epsilon)
                omega[n] += omega_add
                print('parameter :\t', n, '\nomega :\t', omega[n])
                W[n] = p.data.clone().zero_()
    return omega

def update_W(model, W, p_old, args):
    for n, p in model.named_parameters():
        if p.requires_grad and (n.find('bn')==-1):
            n = n.replace('.', '__')
            if p.grad is not None:
                W[n].add_(-p.grad*(p.detach()-p_old[n]))
            p_old[n] = p.detach().clone()
               

def SI_loss(model, omega, prev_params, si_lambda):
    losses = []
    for n, p in model.named_parameters():
        if p.requires_grad and (n.find('bn')==-1):
            n = n.replace('.', '__')
            losses.append((omega[n] * (p - prev_params[n])**2).sum())
    return si_lambda*sum(losses)

def optimizer_utils(args, model, lrs, meta, task_idx):
    if args.si:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.decay)
    elif not(args.si) and args.optim=='Adam':
        if args.ann_meta or args.meta!=0:
            optimizer = Adam_meta(model.parameters(), lr = lrs[task_idx], meta = meta, weight_decay = 0, meta_func=args.meta_ac)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    elif not(args.si) and args.optim=='SGD':
        if args.ann_meta or args.meta!=0:
            optimizer = SGD_meta(model.parameters(), lr=lrs[task_idx], meta = meta, momentum=0, weight_decay=args.decay, meta_func=args.meta_ac)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lrs[task_idx], momentum=0, weight_decay=args.decay)

    return optimizer

def get_gate(args, gate, task_idx, device, phase='train'):
    if args.gate and args.xdg:
        if phase=='train' or 'test':
            gate_tsk = [torch.tensor(gate[task_idx][:,0]).to(device).float(),torch.tensor(gate[task_idx][:,1]).to(device).float()]
        elif phase=='test' and args.test_one_class:
            idx_=task_idx//args.class_per_task
            gate_tsk = [torch.tensor(gate[idx_][:,0]).to(device).float(),torch.tensor(gate[idx_][:,1]).to(device).float()]
    elif args.gate and gate!=[]:
        gate_tsk=[gate[0][task_idx,:],gate[1][task_idx,:]]
    else:
        gate_tsk = []

    return gate_tsk

def meta_generate(args, model, device):
    meta={}
    for n, p in model.named_parameters():
        p.newname = None
        if args.ann_meta and ('fc' in n) or ('cv' in n):
            index = int(n[9])
            p.newname = 'l'+str(index)
            if index<3:
                temp = list(args.meta*np.ones((args.hidden_layers[0],)))
            else:
                temp = list(args.meta*np.ones((args.out_size,)))
            meta[p.newname] = torch.tensor(temp).float().to(device)
    return meta

import random
def gate_load(args, device, gate_prob=0.5):
    if args.gate and args.xdg:
        hidden_neuron=args.hidden_layers[0]
        hidden_layers=len(args.hidden_layers)
        choose_neuron = [int(hidden_neuron*gate_prob) for i in range(len(args.task_sequence))]
        gate=np.zeros([len(args.task_sequence),hidden_neuron,2])
        for i in range(len(args.task_sequence)):
            for j in range(hidden_layers):
                tmp=random.sample(list(range(0,hidden_neuron)),choose_neuron[i])
                gate[i,tmp,j]=1
        print('The probability of gate=1 is', np.sum(gate[0,:,0]==1)/len(gate[0,:,0]))
    elif args.gate:
        if args.class_per_task==1:
            meta_ann_10 = np.load(args.meta_path+'/modulation_signal_for_training_SNN.npy',allow_pickle=True)
        else:
            # print(len(args.task_sequence))
            meta_ann_10 = np.load('modulation_signal_{0}_{1}task_for_training_SNN.npy'.format(args.dataset,len(args.task_sequence)),allow_pickle=True)
        meta_ann=np.argmax(meta_ann_10,axis=2)
        gate = [torch.tensor(~np.in1d(meta_ann[:,i*args.hidden_layers[0]:(i+1)*args.hidden_layers[0]], np.where(np.isinf(args.m_th)))).to(device).reshape(meta_ann.shape[0],args.hidden_layers[0]) for i in range(2)]
    else:
        gate=[]
    return gate

def data_initial(args, test_loader_list):
    data = {}
    data['net'] = args.net
    data['norm'] = args.norm
    data['lr'], data['task_order'] = [], []
    data['ewc'], data['SI'] = [], []
    data['epoch'], data['acc_tr_tag'], data['loss_tr'] = [], [], []

    for i in range(len(test_loader_list)):
        data['acc_test_tsk_'+str(i+1)], data['loss_test_tsk_'+str(i+1)] = [], []

    return data