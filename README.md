An implementation of StyleGAN on CelebA dataset.

## Install 

```shell script
git clone https://github.com/MorvanZhou/celebA-styleGAN
cd celebA-styleGAN
pip3 install -r requirements.txt
```

## Train

```shell script
python3 train.py -b 16 --epoch 51 -lr 0.001 -b1 0. -b2 0.9
```

## Results

First few of epoch:

![](demo/ep000t005000.png)

After one day:

![](demo/ep010t014000.png)
