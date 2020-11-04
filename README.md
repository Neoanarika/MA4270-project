# MA4270 Project 

The code is based on the OnlineNaturalGradient object in Kaldi src/nnet3/natural-gradient-online.h [12]and the github repohttps://github.com/YiwenShaoStephen/NGD-SGD. The main changes that were madewere to allow the program to run on a machine with multiple GTX 2080 Ti for efficient parallel training.

## Notable Files
1. normal.ipynb : generates the pair of normal distributions
2. ngd.py: Is the pytorch implementation of NGD-SGD based on "Parallel training of DNNs with Natural Gradient and Parameter Averaging" by D. Povey, X. Zhang and S. Khudanpur, ICLR Workshop, 2015 and implemented by YiwenSHaoStephen. 
3. cifar.py: CIFAR Model

## How To Run The Code
```
python ./cifar.py \
       --exp exp/cifar/wrn-28-10-ngd \
       -a wrn \
       --depth 28 \
       --widen-factor 10 \
       --optimizer ngd \
       --epochs 50 \
       --scheduler step \
       --milestones 38 \
       --gamma 0.1 \
       --wd 1e-4
```

## How To Run Tensorboard 
```
tensorboard --logdir exp/cifar/exp2
```
