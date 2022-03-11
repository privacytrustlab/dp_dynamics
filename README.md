# Privacy Dynamics for Noisy (Stochastic) Mini-batch Gradient Descent

This repository contains the code to train differentially private models using privacy dynamics analysis. Currently, privacy dynamics analysis supports regularized logistic regression model trained with vanilla noisy mini-batch GD (i.e. no momentum, no adaptive step-size etc.). More details about the algorithm and privacy dynamics analysis are described in our paper:

_Differentially Private Learning Needs Hidden State (Or Much Faster Convergence)_</br>
**Jiayuan Ye and Reza Shokri**</br>
[arXiv:2203.05363](https://arxiv.org/abs/2203.05363)



## Installation

The current code is adapted from the [Handcrafted-DP] repo, with slight modification to the code and requirements. The code was tested with `python 3.8`, `torch 1.10.1` and `CUDA 11.4` using GeForce 3090. To run the code, please set up the environment with the following steps.

1. Create conda environment
```
conda create --name mytf python==3.8.10
conda activate mytf
```

2. Install Cuda Toolkit, CuDNN and pip
```
conda install -c anaconda cudatoolkit
conda install -c conda-forge cudnn
conda install -n mytf pip
```

3. Install pytorch and other requirements with:
```
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install -r requirements.txt
```

The code was tested with `python 3.8`, `torch 1.10.1`, `tensorflow-2.5.0` and `CUDA 11.4`.


<!-- P.S. If you encounter the error: Could not load dynamic library 'libcusolver.so.11' , try to locate the file in installation path and add it
```
find / -name libcusolver.so*
cd /home/ubuntu/anaconda3/envs/mytf/lib/
cp libcusolver.so.10 libcusolver.so.11
``` -->


## Linear models trained on hand-crafted features

This table presents the main results from our paper. We show results for linear models (logistic regression model) trained on "handcrafted" ScatterNet features from scratch.


We compare the privacy accuracy tradeoff of model under following types of algorithms and privacy analyses.
1. (reproduced from [3]) noisy mini-batch GD with momentum and per-example gradient clipping, without regularization (DP-SGD)
2. noisy mini-batch GD with momentum and per-example feature clipping, without regularization (DP-SGD)
3. noisy mini-batch GD with momentum and per-example gradient clipping, with l2 regularization (DP-SGD)
4. noisy mini-batch GD with momentum and per-example feature clipping, with l2 regularization (DP-SGD)
5. vanilla noisy mini-batch GD without momentum, with per-example gradient clipping and per-example feature clipping, with l2 regularization (privacy dynamics analysis and DP-SGD analysis)

We also add the non-dp experiment results for comparison.


For each dataset, we target a privacy budget of `(epsilon=3, delta=10^-5)`.

|Dataset|DP-SGD + momentum (grad. clip.)| DP-SGD + momentum (feature clip.)| DP-SGD + momentum + l2 reg. (grad. clip.) | DP-SGD + momentum + l2 reg. (feature clip.) | DP Dynamics + l2 reg. (grad. clip. + feature clip.)| Non-DP + sgd (no clipping)|
|:--|:--|:--|:--|:--|:--|:--|
|MNIST|98.51%|97.84%|98.48%|97.84%|98.95%|99.3%|
|Fashion-MNIST|89.61%|87.85%|89.53%|87.71%|89.95%|91.5%|
|CIFAR-10|66.75%|63.29%|66.51%|63.26%|69.30%|71.1%|
|Reference|[3]|-|-|-|This paper|[3]|

To reproduce the result for linear ScatterNet models trained using DP-SGD, with momentum 0.9 and gradient clipping, as described in [3], run
```
python3 baselines.py --dataset=mnist --batch_size=4096 --lr=8 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=3.04 --epochs=40
python3 baselines.py --dataset=fmnist --batch_size=8192 --lr=16 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=4.05 --epochs=40
python3 baselines.py --dataset=cifar10 --batch_size=8192 --lr=4 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=5.67 --epochs=60
```

For linear ScatterNet models trained using DP-SGD with momentum 0.9, without regularization, and with feature clipping, run
```
python3 baselines_feature_clip.py --dataset=mnist --batch_size=4096 --lr=8 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=3.04 --max_data_norm=1 --max_grad_norm=2 --epochs=40
python3 baselines_feature_clip.py --dataset=fmnist --batch_size=8192 --lr=16 --epochs=40 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=4.05 --max_data_norm=1 --max_grad_norm=2
python3 baselines_feature_clip.py --dataset=cifar10 --batch_size=8192 --lr=4 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=5.67 --epochs=60 --max_data_norm=1 --max_grad_norm=2
```

For linear ScatterNet models trained using DP-SGD with momentum 0.9, l2 regularization, and with gradient clipping, run
```
python3 baselines.py --dataset=mnist --batch_size=4096 --lr=8 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=3.04 --l2_reg=0.04 --epochs=40
python3 baselines.py --dataset=fmnist --batch_size=8192 --lr=16 --epochs=40 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=4.05 --l2_reg=0.04
python3 baselines.py --dataset=cifar10 --batch_size=8192 --lr=4 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=5.67 --epochs=60 --l2_reg=0.04
```

For linear ScatterNet models trained using DP-SGD with momentum 0.9, l2 regularization, and with feature clipping, run
```
python3 baselines_feature_clip.py --dataset=mnist --batch_size=4096 --lr=8 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=3.04 --epochs=40 --l2_reg=0.04 --max_data_norm=1 --max_grad_norm=2
python3 baselines_feature_clip.py --dataset=fmnist --batch_size=8192 --lr=16 --epochs=40 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=4.05 --l2_reg=0.04 --max_data_norm=1 --max_grad_norm=2
python3 baselines_feature_clip.py --dataset=cifar10 --batch_size=8192 --lr=4 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=5.67 --epochs=60 --max_data_norm=1 --max_grad_norm=2 --l2_reg=0.04
```

For linear ScatterNet models trained using our pivacy dynamics analysis without momentum, with regularization, and with gradient clipping and feature clipping, run
```
python3 baselines_feature_clip.py --dataset=mnist --batch_size=2048 --lr=0.75 --momentum=0 --input_norm=BN --bn_noise_multiplier=8 --noise_multiplier=3.08 --epochs=1200 --l2_reg=0.08 --max_data_norm=2 --max_grad_norm=3.16
python3 baselines_feature_clip.py --dataset=fmnist --batch_size=2048 --lr=1.92 --momentum=0 --input_norm=GroupNorm --num_groups=27 --noise_multiplier=3.01 --epochs=1200 --l2_reg=0.02 --max_data_norm=1 --max_grad_norm=2
python3 baselines_feature_clip.py --dataset=cifar10 --batch_size=2048 --lr=0.75 --momentum=0 --input_norm=BN --bn_noise_multiplier=6 --noise_multiplier=3.23 --epochs=1200 --l2_reg=0.08 --max_data_norm=2 --max_grad_norm=1.58
```

## Linear models fine-tuned using pre-trained models

Public source models fined tuned to do private learning on CIFAR-10
    1. ResNeXt-29 (CIFAR-100)
    2. SIMCLR v2 (unlabelled ImageNet)

The privacy guarantee is fixed to be `(epsilon = 2, delta = 10^-5)`. The fine-tuning is done by running noisy mini-batch GD training on features extracted from the penultimate layer of the source models. We show the transfer accuracy on CIFAR-10 for fine-tuned models as follows.

| Source Model|DP-SGD + momentum (grad. clip.)| DP Dynamics + l2 reg. (grad. clip. + feature clip.)| Non-DP + sgd (no clipping)|
|:--|:--|:--|:--|
| ResNeXt-29 (CIFAR-100)|79.6%|79.85%|84|
| SIMCLR v2(unlabelled ImageNet)|92.4%|92.17%|95.3%|
| Reference | [3] | This paper | [3]|


To extract features frome the source models, first download the `resnext-8x64d` model from [here](https://github.com/bearpaw/pytorch-classification), and then run:
```
python3 -m transfer.extract_cifar100
python3 -m transfer.extract_simclr
```

To reproduce the results in [3] for training linear models with DP-SGD on extracted features, run: 
```
python3 -m transfer.transfer_cifar --feature_path=transfer/features/cifar100_resnext --batch_size=2048 --lr=8 --noise_multiplier=3.32
python3 -m transfer.transfer_cifar --feature_path=transfer/features/simclr_r50_2x_sk1 --batch_size=1024 --lr=4 --noise_multiplier=2.40
```


To train linear models with our privacy dynamics analysis on extracted features, run:
```
python3 -m transfer.transfer_cifar --feature_path=transfer/features/cifar100_resnext --batch_size=1024 --lr=1.92 --noise_multiplier=4.20 --max_data_norm=1 --max_grad_norm=1.5 --l2_reg=0.02 --momentum=0 --epochs=600
python3 -m transfer.transfer_cifar --feature_path=transfer/features/simclr_r50_2x_sk1 --batch_size=1024 --lr=1.92 --noise_multiplier=4.20 --max_data_norm=1 --max_grad_norm=1.5 --l2_reg=0.02 --momentum=0 --epochs=600
```

## Citation

[1] Jiayuan Ye and Reza Shokri. [Differentially Private Learning Needs Hidden State (Or Much Faster Convergence)](https://arxiv.org/abs/2203.05363). arXiv:2203.05363, 2022.

[2] Rishav Chourasia*, Jiayuan Ye*, and Reza Shokri. [Differential Privacy Dynamics of Langevin Diffusion and Noisy Gradient Descent](https://arxiv.org/pdf/2102.05855.pdf). NeurIPS, 2021.

[3] Florian Tramèr and Dan Boneh. [Differentially Private Learning Needs Better Features (or Much More Data)](https://arxiv.org/pdf/2011.11660.pdf). ICLR, 2021.