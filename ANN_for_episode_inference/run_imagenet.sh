# root_path="exp/imagenet2cifar100_100class_pretrain_64ch"
# python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 100
# python visualize_cifar100_gate.py --path $root_path

# root_path="exp/imagenet2cifar100_200class_pretrain_64ch"
# python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 200
# python visualize_cifar100_gate.py --path $root_path

# root_path="exp/imagenet2cifar100_400class_pretrain_64ch"
# python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 400
# python visualize_cifar100_gate.py --path $root_path

# root_path="exp/imagenet2cifar100_600class_pretrain_64ch"
# python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 600
# python visualize_cifar100_gate.py --path $root_path

root_path="exp/imagenet2cifar100_950class_pretrain_64ch"
python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 950 --dataset "imagenet"
python visualize_cifar100_gate.py --path $root_path

# root_path="exp/imagenet2cifar100_900class_pretrain_64ch"
# python train_ann_cifar100_enhanced.py --root_path $root_path  --train_task_num 900
# python visualize_cifar100_gate.py --path $root_path