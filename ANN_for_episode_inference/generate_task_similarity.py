import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch


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


inputs_file_name = 'embeddings/cifar100_clip_vitl14/train/inputs.npy'
labels_file_name = 'embeddings/cifar100_clip_vitl14/train/labels.npy'

inputs = np.load(inputs_file_name).astype(np.float32)
labels = np.load(labels_file_name)
print(inputs.shape)

index = np.argsort(labels)
inputs = inputs[index]
labels = labels[index]


pca = PCA(n_components=500)
pca.fit(inputs)
inputs = pca.transform(inputs[:10000])
print(inputs.shape)

inputs = inputs.reshape(100, 500, -1)
inputs = np.mean(inputs, axis=1)

cor = np.corrcoef(inputs)
cor = np.abs(cor)

plt.matshow(cor, cmap='jet')
plt.colorbar()
plt.savefig('cor_class1.png', dpi=300)
plt.close()

np.save('cor/vitl14.npy', cor)
