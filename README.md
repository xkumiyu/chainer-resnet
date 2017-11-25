# Chainer ResNet

Chainer implementation of [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

# Implemented layers

* for insize = 224 (ex. imagenet)
  * 18, 34, 50, 101, 152
* for insize = 34 (ex. cifar)
  * 20, 32, 44, 56, 110

# Example for training cifar10 dataset

```
$ python train_cifar -d cifar10 -l <layers>
```
