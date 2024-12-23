import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def save_dict(dict, path):
    with open(path, 'wb') as file:
        pickle.dump(dict, file)
    print('>>>>>>>>>>>>>>>>>>dict saved')


def load_dict(path):
    with open(path, 'rb') as file:
        dict = pickle.load(file)
        print('>>>>>>>>>>>>>>>>>dict loaded')
        return dict


def show_picture(tensor):
    array_ = tensor.cpu().numpy()
    array_ = array_.reshape(28, 28)
    plt.imshow(array_, cmap='gray', interpolation='nearest')
    plt.show()


def plot_acc(paths):
    for path in paths:
        dict = load_dict(path)
        acc = dict['acc']
        print(path, acc)
        mean_acc = [np.mean(item) for item in acc]
        plt.plot(mean_acc)
    plt.legend(paths)
    plt.grid()
    plt.savefig('plot.png')


def load_mean(paths):
    for path in paths:
        dict = load_dict(path)
        acc = dict['acc']
        print(len(acc), np.mean(acc[-1]))


# def normalized_hamming_distance(x, y):
#     return np.array(x == y).astype(np.float32).mean()


# def get_cosine_similarity_1dim(gate_1, gate_2):
#     gate = torch.cat((gate_1, gate_2), dim=0)
#     return get_cosine_similarity(gate)


# def get_task_similarity_matrix(index1, index2, size):
#     Y = normalized_hamming_distance(index1, index2)
#     temp = torch.ones([size, size]).cuda()
#     result_1 = torch.cat((temp, Y * temp), dim=1)
#     result_2 = torch.cat((Y * temp, temp), dim=1)
#     result = torch.cat([result_1, result_2], dim=0)
#     return 1 - result

def get_cosine_similarity_1dim(gate):
    eps = 1e-10
    inner_product = gate.mm(gate.t())
    f_norm = torch.norm(gate, p='fro', dim=1, keepdim=True)
    outter_product = f_norm.mm(f_norm.t())
    cosine = inner_product / (outter_product + eps)
    return 1 - cosine


def get_conjugate_loss(gate, sim_gt, l_order):
    loss_list = []
    for i in range(gate.size(-1)):
        sim_i = get_cosine_similarity_1dim(
            gate[:, :, i]).unsqueeze(0)
        
        weight=torch.ones_like(sim_gt).cuda()
        # weight[sim_gt<0.15]=100
        # weight=weight/weight.mean()
        
        loss_list.append(weight*(torch.abs(sim_i-sim_gt)**l_order).mean())
    loss = torch.stack(loss_list)
    loss = loss.mean()
    return loss


def get_dist_loss(gate, expected_dist):
    gate = gate.mean(dim=[0, 1])
    rlt = torch.abs(gate-expected_dist).mean()
    return rlt


# def show_task_similarity(index, root_path):
#     task_num = index.shape[0]
#     result = np.zeros([task_num]*2)
#     for i in range(task_num):
#         for j in range(i, task_num):
#             result[i, j] = normalized_hamming_distance(index[i], index[j])
#             result[j, i] = result[i, j]
#     plt.imshow(result[:100, :100], interpolation='nearest', cmap='jet')
#     plt.colorbar()
#     plt.savefig(root_path+'/task_similarity.png')
#     return result
