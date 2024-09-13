import torch
import torch.nn as nn


class aux_net(nn.Module):
    def __init__(self, inputs_size, hidden_size_list, channels, n_value):
        super(aux_net, self).__init__()
        self.n_value = n_value
        self.linear_1 = nn.Linear(inputs_size, channels)
        self.linear_2 = nn.Linear(channels, channels)

        self.heads = []
        for i in range(len(hidden_size_list)):
            self.heads.append(
                nn.Linear(channels, hidden_size_list[i]*n_value))
        self.heads = nn.ModuleList(self.heads)

        # self.input_norm = nn.BatchNorm1d(inputs_size, affine=False)

    def forward(self, x, scale):

        # x = self.input_norm(x)

        temp = self.linear_1(x)
        temp = torch.relu(temp)

        temp = self.linear_2(temp)
        temp = torch.relu(temp)


        heads = []
        for head in self.heads:
            buf = head(temp)
            buf = buf.reshape(buf.size(0), -1, self.n_value)
            buf = torch.softmax(buf*scale, dim=2)
            heads.append(buf)
        return heads

import math
class Adam_meta(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), meta = {}, eps=1e-8,
                 weight_decay=0, meta_func='tanh', amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, meta=meta, eps=eps,
                        weight_decay=weight_decay, meta_func=meta_func, amsgrad=amsgrad)
        self.meta_func = meta_func
        super(Adam_meta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_meta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:#denom:Vt
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                

                binary_weight_before_update = torch.sign(p.data)
                condition_consolidation = (torch.mul(binary_weight_before_update, exp_avg) > 0.0 )   # exp_avg has the same sign as exp_avg/denom
                if p.dim()==1 or p.dim()==3: # True if p is bias, false if p is weight
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    if self.meta_func=='tanh':
                        tmp_=group['meta'][p.newname]*torch.abs(p.data.transpose(1,0))
                        tmp=torch.pow(torch.tanh(tmp_.transpose(1,0)),2)
                        decayed_exp_avg = torch.mul(torch.ones_like(p.data)-tmp, exp_avg)
                        p.data.addcdiv_(torch.where(condition_consolidation, decayed_exp_avg, exp_avg), denom, value=-step_size)
                    elif self.meta_func=='sigmoid':
                        tmp_=torch.abs(group['meta'][p.newname]*p.data.transpose(1,0)).transpose(1,0)
                        tmp=torch.ones_like(p.data)*2/(torch.ones_like(p.data)+torch.exp(tmp_))
                        decayed_exp_avg = torch.mul(tmp, exp_avg)
                        p.data.addcdiv_(torch.where(condition_consolidation, decayed_exp_avg, exp_avg), denom, value=-step_size)
                    elif self.meta_func=='exp':
                        # print('meta:', group['meta'][p.newname])
                        tmp_ = torch.abs(group['meta'][p.newname]*p.data.transpose(1,0)).transpose(1,0)
                        tmp = torch.exp(-tmp_)
                        # print('meta_exp:', tmp)
                        decayed_exp_avg = torch.mul(tmp, exp_avg)
                        # decayed_exp_avg = torch.mul(tmp, exp_avg)
                        # print('before_detect:', decayed_exp_avg, 'after_detect:', meta_detect(decayed_exp_avg))
                        p.data.addcdiv_(torch.where(condition_consolidation, decayed_exp_avg, exp_avg), denom, value=-step_size)

        return loss