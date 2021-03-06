# MutexMatch4SSL

Code for MutexMatch.

arxiv: [https://arxiv.org/abs/2203.14316][a].
## Requirements
- matplotlib==3.3.2
- numpy==1.19.2
- pandas==1.1.5
- Pillow==9.0.1
- torch==1.4.0+cu92
- torchvision==0.5.0+cu92
## Training
### Important Args
- `--k` : Control the intensity of consistency regularization on TNC. By default, k=num_classes
- `--num_classes` : The number of classes in your dataset.
- `--num_labels` : Amount of labeled data used.  
- `--net_from_name` and `--net` : By default, wide resnet (WRN-28-2) are used for experiments. If you want to use other backbones for tarining, set `--net_from_name True --net @backbone`. We provide alternatives as follows: resnet18, cnn13 and preresnet.
- `--dataset [cifar10/cifar100/svhn/stl10/miniimage]` and `--data` : Your dataset name and path. We support five datasets: CIFAR-10, CIFAR-100, SVHN, STL-10 and mini-ImageNet. When `--dataset stl10`, set `--fold [0/1/2/3/4].`
- `--num_eval_iter` : After how many iterations, we evaluate the model. Note that although we show the accuracy of pseudo-labels on unlabeled data in the evaluation, this is only to show the training process. We did not use any information about labels for unlabeled data in the training. Additionally, when you train model on STL-10, the pseudo-label accuracy will not be displayed normally, because we don't have ground-truth of unlabeled data.
### Training with Single GPU

```
python train_mutex.py --rank 0 --gpu [0/1/...] @@@other args@@@
```
### Training with Multi-GPUs (DistributedDataParallel)

```
## for one node
python train_mutex.py --world-size 1 --rank 0 --multiprocessing-distributed @@@other args@@@
```
### Examples of Running
This code assumes 1 epoch of training, but the number of iterations is 2\*\*20. For CIFAR-100, you need set `--widen_factor 8` for WRN-28-8 whereas WRN-28-2 is used for CIFAR-10.  Note that you need set `--net_from_name True --net resnet18` for STL-10 and mini-ImageNet. 

#### WideResNet

```
## CIFAR-10
python train_mutex.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40  --gpu 0
```

> CIFAR-10, with 40 labels, result of seed 1 (Acc/%): 94.91, weight: [here][cifar10]

***

```
## CIFAR-100 k=60
python train_mutex.py --world-size 1 --rank 0 --lr_decay cos --k 60 --widen_factor 8 --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar100 --dataset cifar100 --num_classes 100 --num_labels 200  --gpu 0
```

> CIFAR-100, with 200 labels, result of seed 1 (Acc/%): 43.84, weight: [here][cifar100]

***

```
## SVHN
python train_mutex.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name svhn --dataset svhn --num_classes 10 --num_labels 40  --gpu 0
```

> SVHN, with 40 labels, result of seed 1 (Acc/%): 97.24, weight: [here][2]

***
#### CNN-13

```
## CIFAR-10
python train_mutex.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 1000 --net_from_name True --net cnn13 --gpu 0
```
> CIFAR-10, with 1000 labels, result of seed 1 (Acc/%): 93.01, weight: [here][3]

***

#### ResNet-18

```
## mini-ImageNet
python train_mutex.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name miniimage --dataset miniimage --num_classes 100 --num_labels 1000 --net_from_name True --net resnet18 --gpu 0
```
> mini-ImageNet, with 1000 labels, result of seed 1 (Acc/%): 47.90, weight: [here][mini]

***
## Resume Training and Evaluation
### Resume
If you restart the training, please use `--resume --load_path @your_path`. 

### Evaluation
```
python eval_mutex.py --data_dir @your_dataset_path --load_path @your_path --dataset @[cifar10/cifar100/svhn/stl10/miniimage] 
```
Use `--net_from_name True` and `--net [cnn13/resnet18]` for different backbones.

## Results (e.g., CIFAR-10)
- k=num_classes

|seed | 10 labels | 20 labels| 40 labels|80 labels|
| :-----:| :-----:| :----: | :----: |:----: |
| 1| 15.73 | 93.43 |94.91 |93.69|
| 2| 71.47 | 93.24 |94.76 |93.64|
| 3| 93.07 | 93.42 |92.96 |92.05|
| 4| 86.32 | 87.56 |88.41 |93.43|
| 5| 65.66 |91.18 |95.05 |93.32|
|avg | 66.45 |91.77 |93.22 |93.23|

- k=0.6\*num_classes

|seed | 10 labels | 20 labels| 40 labels|80 labels|
| :-----:| :-----:| :----: | :----: |:----: |
| 1| 65.01 | 93.85 |94.50 |94.77|
| 2| 19.99 | 92.95 |94.01 |94.67|
| 3| 70.46 | 92.86 |94.74 |94.83|
| 4| 68.89   | 86.63 |94.95 |95.43|
| 5| 63.23 |94.84 |92.84 |95.30|
|avg | 57.52 |92.23 |94.21 |95.00|


## Acknowledgement
Our code is based on open source code: [LeeDoYup/FixMatch-pytorch][1]

[1]: https://github.com/LeeDoYup/FixMatch-pytorch
[2]: https://1drv.ms/u/s!Ao848hI985sshh1L1hbkwSWz7fdu?e=JnFxBB
[3]: https://1drv.ms/u/s!Ao848hI985sshhsvuQSFJ-pu1gRv?e=od6PnI
[cifar10]: https://1drv.ms/u/s!Ao848hI985sshhl8PY0R-xZ-leSu?e=4MPVya
[mini]: https://1drv.ms/u/s!Ao848hI985sshh_x8vW7gACP4SRK?e=iuiici
[cifar100]: https://1drv.ms/u/s!Ao848hI985sshiHv6ghquy7ApJ-_?e=gGOXfh
[a]: https://arxiv.org/abs/2203.14316
