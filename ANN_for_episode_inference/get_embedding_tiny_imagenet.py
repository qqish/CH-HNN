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

from tqdm import tqdm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='ANN training')


parser.add_argument('--batch_size', default=100, type=int)
parser.add_argument('--seed', default=430, type=int)
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)


def get_data_set():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root='./data/tiny-ImageNet-200/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    # test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    test_loader = None

    return train_loader, test_loader


def train_ann_epoch(model, preprocess, data_loader):
    embeddings_list = []
    labels_list = []

    def unfold(inputs_batch):
        inputs_batch = [inputs_batch[i:i+1] for i in range(len(inputs_batch))]
        rlt = []
        for inputs in inputs_batch:

            inputs = torch.nn.Upsample((224, 224))(inputs)
            inputs = inputs.detach().numpy()
            inputs = (inputs*255.).astype(np.uint8)[0]
            inputs = inputs.transpose(1, 2, 0)
            inputs = Image.fromarray(inputs)
            inputs = preprocess(inputs).unsqueeze(0).cuda()
            rlt.append(inputs)
        rlt = torch.cat(rlt, dim=0)
        return rlt

    for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader)):
        inputs = unfold(inputs)

        with torch.no_grad():
            embedding = model.encode_image(inputs)
            embedding = embedding.cpu().detach().numpy()
            embeddings_list.append(embedding)
            labels_list.append(targets.detach().numpy())
    embeddings_list = np.concatenate(embeddings_list, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    np.save('./embeddings/embeddings_tiny_imagenet.npy', embeddings_list)
    np.save('./embeddings/labels_tiny_imagenet.npy', labels_list)
    print(embeddings_list.shape)
    print(labels_list.shape)


def get_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess


def get_ann():
    model = torchvision.models.resnet50(pretrained=True).cuda()
    model.eval()
    model.fc = torch.nn.Identity()
    return model


train_loader, test_loader = get_data_set()
model = get_ann()


train_ann_epoch(model, None, train_loader)
