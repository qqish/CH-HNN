import torch.optim as optim
import numpy as np
import os
import torchvision
import torchvision.transforms as transforms
import argparse
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch.functional as F
import clip
from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='ANN training')

parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--seed', default=430, type=int)
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)


def get_data_set(transform):
    # train_dataset = torchvision.datasets.ImageFolder(
    #     root='./data/tiny-ImageNet-200/train', transform=transform)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True)

    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


def train_ann_epoch(model, preprocess, data_loader, save_path):
    embeddings_list = []
    labels_list = []

    for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader)):

        inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            embedding = model.encode_image(inputs)
            embedding = embedding.cpu().detach().numpy()
            embeddings_list.append(embedding)
            labels_list.append(targets.cpu().detach().numpy())

    embeddings_list = np.concatenate(embeddings_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    np.save(save_path+'inputs.npy', embeddings_list)
    np.save(save_path+'labels.npy', labels_list)
    print(embeddings_list.shape)
    print(labels_list.shape)


def get_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    # model, preprocess = clip.load("RN50", device=device)
    return model, preprocess


model, preprocess = get_clip()
train_loader, test_loader = get_data_set(preprocess)

train_ann_epoch(model, preprocess, train_loader,
                save_path='./embeddings/cifar100_clip_vitl14/train/')
train_ann_epoch(model, preprocess, test_loader,
                save_path='./embeddings/cifar100_clip_vitl14/test/')
