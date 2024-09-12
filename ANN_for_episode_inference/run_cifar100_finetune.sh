root_path="exp/imagenet2cifar100_800class_pretrain_64ch_cifar100_finetune_20class"
python train_ann_cifar100_finetune.py --root_path $root_path  --train_task_num 100 --train_task_ratio 0.2 --dataset "cifar100"
python visualize_cifar100_gate.py --path $root_path

root_path="exp/imagenet2cifar100_800class_pretrain_64ch_cifar100_finetune_10class"
python train_ann_cifar100_finetune.py --root_path $root_path  --train_task_num 100 --train_task_ratio 0.1 --dataset "cifar100"
python visualize_cifar100_gate.py --path $root_path

root_path="exp/imagenet2cifar100_800class_pretrain_64ch_cifar100_finetune_5class"
python train_ann_cifar100_finetune.py --root_path $root_path  --train_task_num 100 --train_task_ratio 0.05 --dataset "cifar100"
python visualize_cifar100_gate.py --path $root_path

root_path="exp/imagenet2cifar100_800class_pretrain_64ch_cifar100_finetune_40class"
python train_ann_cifar100_finetune.py --root_path $root_path  --train_task_num 100 --train_task_ratio 0.40 --dataset "cifar100"
python visualize_cifar100_gate.py --path $root_path

root_path="exp/imagenet2cifar100_800class_pretrain_64ch_cifar100_finetune_60class"
python train_ann_cifar100_finetune.py --root_path $root_path  --train_task_num 100 --train_task_ratio 0.60 --dataset "cifar100"
python visualize_cifar100_gate.py --path $root_path

root_path="exp/imagenet2cifar100_800class_pretrain_64ch_cifar100_finetune_80class"
python train_ann_cifar100_finetune.py --root_path $root_path  --train_task_num 100 --train_task_ratio 0.80 --dataset "cifar100"
python visualize_cifar100_gate.py --path $root_path


