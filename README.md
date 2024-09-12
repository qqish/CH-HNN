**Code repository for CH-HNN project**, submitted alongside a manuscript to **Nature Communications**.

## Overview
**CH-HNN** is a neural network model designed for **hybrid neural networks for continual learning**, inspired by **cortico-hippocampal circuits**. This repository contains the code necessary to replicate the experiments and analyses described in the associated manuscript.

## Usage
1. For getting the embedding features of **CIFAR-100** and **TinyImageNet**, run `get_embedding_clip.py`.

2. For training Spiking Neural Networks (SNNs) under the guidance of pre-trained Artificial Neural Network (ANNs) within CH-HNN, run the following command:

```bash
python manager.py --ep-inference --dataset 'cifar100' --scenario 'class-incre'
```
We provide pretrained ANN models in the `ANN_Prior` directory, which includes datasets for:

- **sMNIST**
- **pMNIST**
- **CIFAR-100**
- **Tiny-ImageNet**

These datasets can be selected by using the `--dataset` flag.

These datasets are provided for both **class-incremental** and **task-incremental** learning scenarios, which can be selected by using the `--scenario` flag.

Additionally, the `ANN_Prior` directory contains ANN models trained with prior knowledge from **ImageNet** and designed for **continual learning**. These models can also be found in the `ANN_Prior` directory.

3. For testing the class-incremental learning in CIFAR-100, and Tiny-ImageNet dataset, run the following command:
CH-HNN model:
```bash
python demo_test.py --ep-inference --dataset 'cifar100'
```
Baseline model:
```bash
python demo_test.py --dataset 'cifar100'
```

4. For training the ANNs to learn related-episode knowledge, run the following command under the `ANN_for_episode_inference` directory:
```bash
    python manager.py
```
By changing the `_continue`, `_enhanced`, or `_task` tail marker in the `manager.py` file, the ANN can be trained under different designs. 
