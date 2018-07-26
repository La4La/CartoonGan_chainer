import chainer
from chainer import initializers
from chainer import functions as F
from chainer import links as L

import numpy as np


class ResLayer(chainer.Chain):
    def __init__(self, ch, ksize, stride, pad):
        super().__init__()
        initialW = initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels=ch, out_channels=ch, ksize=ksize, stride=stride, pad=pad, initialW=initialW)
            self.bn1 = L.BatchNormalization(ch)
            self.conv2 = L.Convolution2D(in_channels=ch, out_channels=ch, ksize=ksize, stride=stride, pad=pad, initialW=initialW)
            self.bn2 = L.BatchNormalization(ch)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h2 = self.bn2(self.conv2(h1))
        return x + h2


class Block(chainer.ChainList):
    def __init__(self, n_blocks, ch, ksize, stride, pad):
        super().__init__()
        for i in range(n_blocks):
            self.add_link(ResLayer(ch, ksize, stride, pad))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x


class Generator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            initialW = chainer.initializers.HeNormal()
            # Input Convolution
            # All image should be resized and cropped as 3 * 256 * 256
            # input size (batch_size, 3, 256, 256)
            self.in_conv = L.Convolution2D(in_channels=3, out_channels=64, ksize=7, stride=1, pad=3,
                                           initialW=initialW)
            # output size (batch_size, 64, 256, 256)
            self.in_bn = L.BatchNormalization(size=64)

            # Down Convvolution
            self.down_conv1 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=2, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 128, 128, 128)
            self.down_conv2 = L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 128, 128, 128)
            self.down_bn1 = L.BatchNormalization(size=128)

            self.down_conv3 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=2, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 256, 64, 64)
            self.down_conv4 = L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 256, 64, 64)
            self.down_bn2 = L.BatchNormalization(size=256)

            # Residual Blocks
            self.blocks = Block(n_blocks=8, ch=256, ksize=3, stride=1, pad=1)
            # output size (batch_size, 256, 64, 64)

            # Up Convolution
            # ksize=4 used here instead of 3 in the original paper because Chainer doesn't support output padding!
            self.up_conv1 = L.Deconvolution2D(in_channels=256, out_channels=128, ksize=4, stride=2, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 128, 128, 128)
            self.up_conv2 = L.Deconvolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 128, 128, 128)
            self.up_bn1 = L.BatchNormalization(128)

            # ksize=4 used here instead of 3 in the original paper because Chainer doesn't support output padding!
            self.up_conv3 = L.Deconvolution2D(in_channels=128, out_channels=64, ksize=4, stride=2, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 64, 256, 256)
            self.up_conv4 = L.Deconvolution2D(in_channels=64, out_channels=64, ksize=3, stride=1, pad=1,
                                              initialW=initialW)
            # output size (batch_size, 64, 256, 256)
            self.up_bn2 = L.BatchNormalization(64)

            # Output Convolution
            self.out_conv = L.Convolution2D(in_channels=64, out_channels=3, ksize=7, stride=1, pad=3,
                                            initialW=initialW)
            # output size (batch_size, 3, 256, 256)

    def __call__(self, x):
        h = F.relu(self.in_bn(self.in_conv(x)))
        h = F.relu(self.down_bn1(self.down_conv2(self.down_conv1(h))))
        h = F.relu(self.down_bn2(self.down_conv4(self.down_conv3(h))))
        h = self.blocks(h)
        h = F.relu(self.up_bn1(self.up_conv2(self.up_conv1(h))))
        h = F.relu(self.up_bn2(self.up_conv4(self.up_conv3(h))))
        h = self.out_conv(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            initialW = chainer.initializers.HeNormal()
            # Input Convolution
            # All image should be resized and cropped as 3 * 256 * 256
            # input size (batch_size, 3, 256, 256)
            self.in_conv = L.Convolution2D(in_channels=3, out_channels=32, ksize=3, stride=1, pad=1,
                                           initialW=initialW)
            # output size (batch_size, 32, 256, 256)

            self.conv1 = L.Convolution2D(in_channels=32, out_channels=64, ksize=3, stride=2, pad=1,
                                         initialW=initialW)
            # output size (batch_size, 64, 128, 128)

            self.conv2 = L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=1, pad=1,
                                         initialW=initialW)
            # output size (batch_size, 128, 128, 128)
            self.bn1 = L.BatchNormalization(128)

            self.conv3 = L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=2, pad=1,
                                         initialW=initialW)
            # output size (batch_size, 128, 64, 64)

            self.conv4 = L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1, pad=1,
                                         initialW=initialW)
            # output size (batch_size, 256, 64, 64)
            self.bn2 = L.BatchNormalization(256)

            self.conv5 = L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1,
                                         initialW=initialW)
            # output size (batch_size, 256, 64, 64)
            self.bn3 = L.BatchNormalization(256)

            # Output Convolution
            self.out_conv = L.Convolution2D(in_channels=256, out_channels=1, ksize=3, stride=1, pad=1,
                                            initialW=initialW)
            # output size (batch_size, 1, 64, 64)

    def __call__(self, x):
        h = F.leaky_relu(self.in_conv(x))
        h = F.leaky_relu(self.conv1(h))
        h = F.leaky_relu(self.bn1(self.conv2(h)))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.bn2(self.conv4(h)))
        h = F.leaky_relu(self.bn3(self.conv5(h)))
        h = self.out_conv(h)
        return h


class VGG(chainer.Chain):
    def __init__(self):
        super().__init__(
            model = L.VGG16Layers()
        )
        # Pre-trained VGG16 used here instead of VGG19 in original paper
        self._using_layer = "conv4_3"

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 1) * 255
        mean = self.xp.array([103.939, 116.779, 123.68], dtype=np.float32)
        x = x - mean
        x = x.transpose(0, 3, 1, 2)

        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                outputs = self.model(x, layers=[self._using_layer])  # vggの前処理？ modelの部分で -0.5?
                # outputs = [output for output in outputs.values()]
                outputs = outputs[self._using_layer]
        return outputs
