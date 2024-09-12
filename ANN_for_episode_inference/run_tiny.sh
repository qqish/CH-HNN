root_path="exp_pmnist_0827/stage_1_r0.25_gate64"
python train_ann_cifar100_enhanced.py --root_path $root_path --dataset "tinyimagenet" --train_task_num 200
python visualize_cifar100_gate.py --path $root_path