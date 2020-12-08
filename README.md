An implementation of StyleGAN on CelebA dataset.

## Install 

```shell script
git clone https://github.com/MorvanZhou/celebA-styleGAN
cd celebA-styleGAN
pip3 install -r requirements.txt
```

## Train

```shell script
python3 train.py -b 32 --epoch 101 -lr 0.001 -b1 0.2 -b2 0.95
```

## Results

First few of epoch:

![](demo/ep000t005000.png)

After one day:

![](demo/ep010t014000.png)
