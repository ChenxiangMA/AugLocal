# ResNet32 with AugLocal
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 16 --epochs 400  --batch_size 1024 --rule AugLocal --aux_net_depth 1 --pyramid --pyramid_coeff 0.5
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 16 --epochs 400  --batch_size 1024 --rule AugLocal --aux_net_depth 3 --pyramid --pyramid_coeff 0.5
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 16 --epochs 400  --batch_size 1024 --rule AugLocal --aux_net_depth 5 --pyramid --pyramid_coeff 0.5

# ResNet110 with AugLocal
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 55 --epochs 400  --batch_size 1024 --rule AugLocal --aux_net_depth 1 --pyramid --pyramid_coeff 0.5
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 55 --epochs 400  --batch_size 1024 --rule AugLocal --aux_net_depth 3 --pyramid --pyramid_coeff 0.5
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 55 --epochs 400  --batch_size 1024 --rule AugLocal --aux_net_depth 5 --pyramid --pyramid_coeff 0.5

# InfoPro
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 16 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0.05 --ixx_2 0.2 --ixy_2 0.5 --epochs 400  --batch_size 1024 --rule InfoPro
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 55 --aux_net_feature_dim 128 --ixx_1 5 --ixy_1 0   --ixx_2 0.5 --ixy_2 1 --epochs 400  --batch_size 1024 --rule InfoPro

# DGL
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 16 --aux_net_feature_dim 128 --epochs 400  --batch_size 1024 --rule DGL
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 55 --aux_net_feature_dim 128 --epochs 400  --batch_size 1024 --rule DGL


# PredSim
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 16 --aux_net_feature_dim 128 --ixx_1 0.99 --ixy_1 0.01 --ixx_2 0.99 --ixy_2 0.01 --epochs 400  --batch_size 1024 --rule PredSim
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 55 --aux_net_feature_dim 128 --ixx_1 0.99 --ixy_1 0.01 --ixx_2 0.99 --ixy_2 0.01 --epochs 400  --batch_size 1024 --rule PredSim

# BP
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 32 --cos_lr --local_module_num 1 --epochs 400  --batch_size 1024 --rule BP
CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model resnet --layers 110 --cos_lr --local_module_num 1 --epochs 400  --batch_size 1024 --rule BP



