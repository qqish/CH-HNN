import os
from multiprocessing import Process
import time
import numpy as np

gpu_list=[1,2]

def exe(cmd):
    os.system(cmd)

def generate_gate_continue():
    cnt=0
    class_num_all=[[200]]
    task_num_all=[[200]]
    neurons=[64,]
    epochs_all=[60]
    channels_all=[256]
    lr=0.0001
    meta_all=[6,7,9]
    # task_sequence=[0,50,100]
    for i, task_num in enumerate(task_num_all):
        for j, neuron in enumerate(neurons):
            for _, epochs in enumerate(epochs_all):
                for _, class_num in enumerate(class_num_all):
                    for _, c in enumerate(channels_all):
                        for _,meta in enumerate(meta_all):
                            cnt=cnt+1

                            gpu=gpu_list[cnt%len(gpu_list)]
                            cmd='python train_ann_continued.py '
                            cmd+='--gpu {} '.format(gpu)
                            cmd+='--dataset "tinyimage" '
                            cmd+='--stage_1_epoch {} '.format(epochs)
                            cmd+='--meta {} '.format(meta)
                            cmd+='--root_path "./tinyimage_continue_meta{5}_c{3}_epochs_{2}_lr_{4}/exp_gate{1}/" '.format(task_num[0], neuron, epochs, c, lr, meta)
                            cmd+='--n_dim {} '.format(neuron)
                            cmd+='--channels {} '.format(c)
                            cmd+='--train_task_num '

                            for m in range(len(task_num)):
                                cmd+='{} '.format(task_num[m])
                            cmd+='--test_task_num '
                            for m in range(len(class_num)):
                                cmd+='{} '.format(class_num[m])
                            print(cmd)
                            p = Process(target=exe, args=(cmd,))
                            p.start()
                            time.sleep(5)  

def generate_gate_enhanced():
    cnt=0
    class_num_all=[[200]]
    task_num_all=[[200]]
    neurons=[64]
    epochs_all=[60]
    channels_all=[256]
    lr=0.0001
    input_size=768
    # task_sequence=[0,50,100]
    for i, task_num in enumerate(task_num_all):
        for j, neuron in enumerate(neurons):
            for _, epochs in enumerate(epochs_all):
                for _, class_num in enumerate(class_num_all):
                    for _, c in enumerate(channels_all):
                        cnt=cnt+1

                        gpu=gpu_list[cnt%len(gpu_list)]
                        cmd='python train_ann_enhanced.py '
                        cmd+='--gpu {} '.format(gpu)
                        cmd+='--dataset "tinyimage" '
                        cmd+='--lr_s1 {} '.format(lr)
                        cmd+='--stage_1_epoch {} '.format(epochs)
                        # cmd+='--root_path "exp_cifar100_2024_random{0}{2}_based_tinyimage_epochs_{3}/exp_gate{1}/" '.format(task_num[0], neuron, task_num[1], epochs)
                        # cmd+='--root_path "exp_dvsG_random{0}_epochs_{2}_{3}_lr_{4}_100000/exp_gate{1}/" '.format(task_num[0], neuron, epochs, c, lr)
                        cmd+='--root_path "exp_tinyimage_random{0}_epochs_{2}/exp_gate{1}/" '.format(task_num[0], neuron, epochs,)
                        cmd+='--n_dim {} '.format(neuron)
                        cmd+='--channels {} '.format(c)
                        cmd+='--inputs_size {} '.format(input_size)
                        cmd+='--class-num '
                        for m in range(len(class_num)):
                            cmd+='{} '.format(class_num[m])
                        cmd+='--train_task_num '

                        for m in range(len(task_num)):
                            cmd+='{} '.format(task_num[m])
                        cmd+='--test_task_num '
                        for m in range(len(class_num)):
                            cmd+='{} '.format(class_num[m])
                        print(cmd)
                        p = Process(target=exe, args=(cmd,))
                        p.start()
                        time.sleep(5)  

def generate_gate_task():
    cnt=0
    class_num_all=[[200]]
    task_num_all=[[200]]
    neurons=[64]
    epochs_all=[60]
    channels_all=[256]
    lr=0.0001
    dataset='tinyimage'
    input_size=768
    # task_sequence=[0,50,100]
    for i, task_num in enumerate(task_num_all):
        for j, neuron in enumerate(neurons):
            for _, epochs in enumerate(epochs_all):
                for _, class_num in enumerate(class_num_all):
                    for _, c in enumerate(channels_all):
                        cnt=cnt+1

                        gpu=gpu_list[cnt%len(gpu_list)]
                        cmd='python train_ann_task.py '
                        cmd+='--gpu {} '.format(gpu)
                        cmd+='--dataset {} '.format(dataset)
                        cmd+='--lr_s1 {} '.format(lr)
                        cmd+='--stage_1_epoch {} '.format(epochs)
                        # cmd+='--root_path "exp_cifar100_2024_random{0}{2}_based_tinyimage_epochs_{3}/exp_gate{1}/" '.format(task_num[0], neuron, task_num[1], epochs)
                        cmd+='--root_path "task_{3}_random{0}_epochs_{2}/exp_gate{1}/" '.format(task_num[0], neuron, epochs, dataset)
                        cmd+='--n_dim {} '.format(neuron)
                        cmd+='--channels {} '.format(c)
                        cmd+='--inputs_size {} '.format(input_size)
                        cmd+='--train_task_num '

                        for m in range(len(task_num)):
                            cmd+='{} '.format(task_num[m])
                        cmd+='--test_task_num '
                        for m in range(len(class_num)):
                            cmd+='{} '.format(class_num[m])
                        print(cmd)
                        p = Process(target=exe, args=(cmd,))
                        p.start()
                        time.sleep(5)  
generate_gate_continue()