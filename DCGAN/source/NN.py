# coding: utf-8
import chainer
from chainer import Variable,optimizers
import chainer.functions as F
import chainer.links as L
from chainer import Chain
import matplotlib.pyplot as plt
import numpy as np

class Generator(Chain):
    '''ランダムなベクトルから画像を生成する画像作成する'''
    def __init__(self,z_dim):
        super(Generator, self).__init__(
            l1=L.Linear(z_dim,3*3*512),
            dc1 = L.Deconvolution2D(512,256,2,stride=2,pad=1,),
            dc2 = L.Deconvolution2D(256,128,2,stride=2,pad=1,),
            dc3 = L.Deconvolution2D(128,64,2,stride=2,pad=1,),
            dc4 = L.Deconvolution2D(64,1,3,stride=3,pad=1,),

            bn1=L.BatchNormalization(512),
            bn2=L.BatchNormalization(256),
            bn3=L.BatchNormalization(128),
            bn4=L.BatchNormalization(64),

        )
        self.z_dim = z_dim
    def __call__(self, z, test=False):
        h = self.l1(z)
        h = F.reshape(h,(z.data.shape[0],512,3,3))
        h = F.relu(self.bn1(h))
        h = F.relu(self.bn2(self.dc1(h)))
        h = F.relu(self.bn3(self.dc2(h)))
        h = F.relu(self.bn4(self.dc3(h)))
        x = self.dc4(h)
        return x

class Discriminator(Chain):
    def __init__(self, ):
        super(Discriminator, self).__init__(
            c1 = L.Convolution2D(1,64,3,stride=3,pad=1),
            c2 = L.Convolution2D(64,128,2,stride=2,pad=1),
            c3 = L.Convolution2D(128,256,2,stride=2,pad=1),
            c4 = L.Convolution2D(256,512,2,stride=2,pad=1),

            l1 = L.Linear(3*3*512,2),

            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
    def __call__(self, x, test=False):
        """判別関数"""
        h = F.relu(self.c1(x))
        h = F.relu(self.bn1(self.c2(h)))
        h = F.relu(self.bn2(self.c3(h)))
        h = F.relu(self.bn3(self.c4(h)))
        y = self.l1(h)
        return y
