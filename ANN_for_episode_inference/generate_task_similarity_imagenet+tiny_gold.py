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
    cosine_similarity = np.matmul(X, Y.T) / (X_norm * Y_norm+1e-8)
    return cosine_similarity


def get_topk(a, k):
    topk, indices = torch.topk(a, k, dim=1)
    result = torch.zeros_like(a)
    result.scatter_(1, indices, 1)
    return result


inputs_file_name_t200 = 'embeddings/tinyimagenet_clip_vitl14/train/inputs.npy'
labels_file_name_t200 = 'embeddings/tinyimagenet_clip_vitl14/train/labels.npy'

inputs_t200 = np.load(inputs_file_name_t200).astype(np.float32)
labels_t200 = np.load(labels_file_name_t200)
inputs_t200 = inputs_t200/np.linalg.norm(inputs_t200, axis=1, keepdims=True)

labels_t200 += 950

inputs_file_name_i1000 = 'embeddings/imagenet_clip_vitl14_part/train/'
labels_file_name_i1000 = 'embeddings/imagenet_clip_vitl14_part/train/'

inputs_i1000_list = []
labels_i1000_list = []

for i in range(25):
    inputs_i1000 = np.load(inputs_file_name_i1000 +
                           f'inputs_{(i+1)*500}.npy').astype(np.float32)
    labels_i1000 = np.load(labels_file_name_i1000+f'labels_{(i+1)*500}.npy')
    inputs_i1000_list.append(inputs_i1000)
    labels_i1000_list.append(labels_i1000)

inputs_i1000 = np.concatenate(inputs_i1000_list, axis=0)
labels_i1000 = np.concatenate(labels_i1000_list, axis=0)

index = labels_i1000 < 950
inputs_i1000 = inputs_i1000[index]
labels_i1000 = labels_i1000[index]
inputs_i1000 = inputs_i1000/np.linalg.norm(inputs_i1000, axis=1, keepdims=True)


inputs = np.concatenate([inputs_i1000, inputs_t200], axis=0)
labels = np.concatenate([labels_i1000, labels_t200], axis=0)

np.save('embeddings/imagenet+tiny_clip_vitl14/train/inputs.npy', inputs)
np.save('embeddings/imagenet+tiny_clip_vitl14/train/labels.npy', labels)


n_dim = 128
pca = PCA(n_components=n_dim)
pca.fit(inputs)
inputs = pca.transform(inputs)
print('pca', inputs.shape)

inputs_mean_list = []

for i in range(1150):
    inputs_tmp = inputs[labels == i]
    if inputs_tmp.shape[0] == 0:
        print(i)
        raise ValueError
    else:
        inputs_mean_list.append(np.mean(inputs_tmp, axis=0))

inputs = np.array(inputs_mean_list)
print('pca', inputs.shape)


plt.matshow(inputs[:100], cmap='jet')
plt.colorbar()
plt.savefig('pca.png', dpi=300)
plt.close()

# calculate cosine similarity
cosine_similarity = get_cosine_similarity(inputs, inputs)
cor = np.abs(cosine_similarity)

plt.matshow(cor, cmap='jet')
plt.colorbar()
plt.savefig('cor_class_imagenet2cifar100.png', dpi=300)
plt.close()

np.save('cor/imagenet+tiny_vitl14.npy', cor)
