import matplotlib
matplotlib.use('agg')

import torch

from torch.optim.optimizer import required

class SGD_meta(torch.optim.Optimizer):
    def __init__(self, params, lr=required, meta={}, momentum=0, dampening=0,
                 weight_decay=0, meta_func='tanh', nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, meta_func=meta_func, nesterov=nesterov, meta=meta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.meta_func=meta_func
        super(SGD_meta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_meta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # print('p:', p.shape)
                if p.grad is None:
                    continue
                d_p=p.grad
                state = self.state[p]
                if group['weight_decay'] != 0:#param:weight?
                    d_p = d_p.add(p.data, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    if 'momentum_buffer' not in state:
                        buf = torch.clone(d_p).detach()
                        state['momentum_buffer'] = buf
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['dampening'])

                    if group['nesterov']:
                        d_p = d_p.add(buf, alpha=group['momentum'])
                    else:
                        d_p = buf
                binary_weight_before_update = torch.sign(p.data)
                condition_consolidation = (torch.mul(binary_weight_before_update, d_p) > 0.0 )
                step_size=group['lr']
                if p.dim()==1 or p.dim()==3 or p.newname==None or p.shape[0]==50:
                    p.data.add_(d_p, alpha=-step_size)
                else:
                    if self.meta_func == 'tanh':
                        tmp_ = group['meta'][p.newname]*torch.abs(p.data.transpose(1, 0))
                        tmp = torch.ones_like(p.data)-torch.pow(torch.tanh(tmp_.transpose(1, 0)), 2)
                    elif self.meta_func == 'sigmoid':
                        tmp_ = torch.abs(group['meta'][p.newname]*p.data.transpose(1, 0)).transpose(1, 0)
                        tmp = meta_detect(torch.ones_like(p.data)*2/(torch.ones_like(p.data)+torch.exp(tmp_)))
                    elif self.meta_func == 'exp':
                        tmp_ = torch.abs(group['meta'][p.newname] * p.data.transpose(1, 0)).transpose(1, 0)
                        tmp = meta_detect(torch.exp(-tmp_))
                    p.data.add_(torch.where(condition_consolidation, torch.mul(d_p, tmp), d_p), alpha=-step_size)#meta

        return loss

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
        self.meta_func=meta_func
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
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
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
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
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
                        tmp=meta_detect(torch.ones_like(p.data)*2/(torch.ones_like(p.data)+torch.exp(tmp_)))
                        decayed_exp_avg = meta_detect(torch.mul(tmp, exp_avg))
                        p.data.addcdiv_(torch.where(condition_consolidation, decayed_exp_avg, exp_avg), denom, value=-step_size)
                    elif self.meta_func=='exp':
                        tmp_ = torch.abs(group['meta'][p.newname]*p.data.transpose(1,0)).transpose(1,0)
                        tmp = torch.exp(-tmp_)
                        decayed_exp_avg = meta_detect(torch.mul(tmp, exp_avg))
                        p.data.addcdiv_(torch.where(condition_consolidation, decayed_exp_avg, exp_avg), denom, value=-step_size)

        return loss

def meta_detect(x):
    x[torch.isnan(x)]=0
    return x