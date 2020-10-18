# MA4270 Project 

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