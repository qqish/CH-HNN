import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch


# def sparse2coarse(targets):
#     """Convert Pytorch CIFAR100 sparse targets to coarse targets.

#     Usage:
#         trainset = torchvision.datasets.CIFAR100(path)
#         trainset.targets = sparse2coarse(trainset.targets)
#     """
#     coarse_labels = np.array([4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
#                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
#                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
#                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
#                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
#                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
#                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
#                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
#                               16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
#                               18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
#     return coarse_labels[targets]


def get_cosine_similarity(X, Y):
    X_norm = np.linalg.norm(X, axis=1)
    Y_norm = np.linalg.norm(Y, axis=1)
    X_norm = X_norm.reshape(-1, 1)
    Y_norm = Y_norm.reshape(-1, 1)
    X_norm = np.tile(X_norm, (1, Y.shape[0]))
    Y_norm = np.tile(Y_norm, (1, X.shape[0])).T
    cosine_similarity = np.matmul(X, Y.T) / (X_norm * Y_norm)
    return cosine_similarity


def get_topk(a, k):
    topk, indices = torch.topk(a, k, dim=1)
    result = torch.zeros_like(a)
    result.scatter_(1, indices, 1)
    return result


# inputs_file_name_c100 = 'embeddings/cifar100_clip_vitl14/train/inputs.npy'
# labels_file_name_c100 = 'embeddings/cifar100_clip_vitl14/train/labels.npy'


# inputs_c100 = np.load(inputs_file_name_c100).astype(np.float32)
# labels_c100 = np.load(labels_file_name_c100)
# inputs_c100 = inputs_c100/np.linalg.norm(inputs_c100, axis=1, keepdims=True)


# corase_labels = sparse2coarse(labels_c100)
# sort_inputs_list = []
# sort_targets_list = []
# for i in range(100):
#     labels_tmp = labels_c100[corase_labels == i]
#     inputs_tmp = inputs_c100[corase_labels == i]
#     index = np.argsort(labels_tmp)
#     sort_inputs_list.append(inputs_tmp[index])
#     sort_targets_list.append(labels_tmp[index])
# inputs_c100 = np.concatenate(sort_inputs_list, axis=0)
# labels_c100 = np.concatenate(sort_targets_list, axis=0)
# labels_c100 += 200
# print(inputs_c100.shape)
# print(labels_c100.shape)


inputs_file_name_t200 = 'embeddings/tinyimagenet_clip_vitl14/train/inputs.npy'
labels_file_name_t200 = 'embeddings/tinyimagenet_clip_vitl14/train/labels.npy'


inputs_t200 = np.load(inputs_file_name_t200).astype(np.float32)
labels_t200 = np.load(labels_file_name_t200)

inputs_t200 = inputs_t200 / np.linalg.norm(inputs_t200, axis=1, keepdims=True)

index = np.argsort(labels_t200)
inputs_t200 = inputs_t200[index]
labels_t200 = labels_t200[index]

# inputs = np.concatenate([inputs_t200, inputs_c100], axis=0)
# labels = np.concatenate([labels_t200, labels_c100], axis=0)


# index = np.argsort(labels)
# inputs = inputs[index]
# labels = labels[index]


n_dim = 128
pca = PCA(n_components=n_dim)
pca.fit(inputs_t200)
inputs = pca.transform(inputs_t200)

# plt.matshow(inputs[:1000],cmap='jet')
# plt.colorbar()
# plt.savefig('pca.png', dpi=300)
# plt.close()
print(inputs.shape)


inputs = inputs.reshape(-1, 500, n_dim)
print('pca', inputs.shape)
inputs = np.mean(inputs, axis=1)

plt.matshow(inputs[:100], cmap='jet')
plt.colorbar()
plt.savefig('pca.png', dpi=300)
plt.close()

# calculate cosine similarity
cosine_similarity = get_cosine_similarity(inputs, inputs)
cor = np.abs(cosine_similarity)

plt.matshow(cor, cmap='jet')
plt.colorbar()
plt.savefig('cor_class_tiny2cifar100.png', dpi=300)
plt.close()

np.save('cor/tinyimagenet_vitl14.npy', cor)
