root_path="exp/imagenet+tiny2cifar100_1150class_pretrain_64ch"
python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 1150 --dataset "imagenet+tiny"
python visualize_cifar100_gate.py --path $root_path

