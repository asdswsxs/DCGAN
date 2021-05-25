#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import pickle
import random
import sys

argv = sys.argv
argc = len(argv)

# xp = cuda.cupy  ## added
# xp = np
# Set data

train = np.load('books_train.npy')    ### 2000 ＊200
labels = np.load('train_y.npy')
test = np.load('books_test.npy')
gans = np.load('test_y.npy')
unlabeled_data = np.load('books_unlabel.npy')  ### 10000 ＊200

fvar = np.zeros(400).reshape(2,200)
g0 = train[np.where(labels == 0)[0]]
g1 = train[np.where(labels == 1)[0]]
mu0 = np.mean(g0, axis=0)
mu1 = np.mean(g1, axis=0)
v0 = np.sqrt(np.var(g0, axis=0))
v1 = np.sqrt(np.var(g1, axis=0))
fvar[0] = v0
fvar[1] = v1
# gen.var.W[0].data = v0
# gen.var.W[1].data = v1


# Define model

#generator------------------------------------------------------------------

class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            mu = L.EmbedID(2, 200),
        )
    def __call__(self, cls, train=True):
        if (train == True):
            mu0 = self.mu.W[cls]
            z = np.random.rand(200)
            g = z * fvar[cls] + mu0
            return g
        else:
            chainer.config.train = False
            mu0 = self.mu.W[cls]
            z = np.random.rand(200)
            g = x * fvar[cls] + mu0
            chainer.config.train = True
            return g


# class Generator(chainer.Chain):
#     def __init__(self):
#         super(Generator, self).__init__(
#             mu = L.EmbedID(2, 200),
#             var = L.EmbedID(2, 200),
#         )
#     def __call__(self, cls, train=True):
#         if (train == True):
#             mu0 = self.mu.W[cls]
#             var0 = self.var.W[cls]
#             r = np.random.rand(200)
#             g = r * var0 + mu0
#             return g
#         else:
#             chainer.config.train = False
#             mu0 = self.mu.W[cls]
#             var0 = self.var.W[cls]
#             r = np.random.rand(200)
#             g = r * var0 + mu0
#             chainer.config.train = True
#             return g

    # gen.mu.W.shape

#discriminator-----------------------------------------------------------------

class Discriminator(chainer.Chain):   # real = 1, fake = 0
    def __init__(self):
        super(Discriminator, self).__init__(
            l1 = L.Linear(201, 20),
            l2 = L.Linear(20, 2),
        )
    def __call__(self, x, y, train=True):
        if (train == True):
            return F.softmax_cross_entropy(self.fwd(x), y)
        else:
            chainer.config.train = False
            loss = F.softmax_cross_entropy(self.fwd(x), y)
            chainer.config.train = True
            return loss
    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2


#classifier-----------------------------------------------------------------

class Classifier(chainer.Chain):
    def __init__(self):
        super(Classifier, self).__init__(
            l1=L.Linear(200,100),
            l2=L.Linear(100,20),
            l3=L.Linear(20,2),
        )
    def __call__(self, x, y):
        return F.softmax_cross_entropy(self.fwd(x), y)
    def fwd(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3

# functions -----------------------------------------------------------------

def print_test(test):
    xt = Variable(test)
    yy = cla.fwd(xt)
    ans = yy.data
    nrow, ncol = ans.shape
    ok = 0
    for i in range(nrow):
        cls = np.argmax(ans[i,:])
#        print(ans[i,:], cls)
        if cls == gans[i]:
            ok += 1
    print(ok, "/", nrow, " = ", (ok * 1.0)/nrow)


# Initialize model

cla = Classifier()
gen = Generator()
dis = Discriminator()

optimizer_cla = optimizers.Adam(alpha=0.0005)
optimizer_gen = optimizers.Adam()
optimizer_dis = optimizers.Adam(alpha=0.0005)

optimizer_cla.setup(cla)
optimizer_gen.setup(gen)
optimizer_dis.setup(dis)

# Initialize generator

# g0 = train[np.where(labels == 0)[0]]
# g1 = train[np.where(labels == 1)[0]]
# mu0 = np.mean(g0, axis=0)
# mu1 = np.mean(g1, axis=0)
# v0 = np.sqrt(np.var(g0, axis=0))
# v1 = np.sqrt(np.var(g1, axis=0))

# gen.mu.W[0].data = mu0
# gen.mu.W[1].data = mu1
# gen.var.W[0].data = v0
# gen.var.W[1].data = v1

# Learn

# n = len(train)  ## 20000
n1 = 2000
n2 = 10000
bs1 = 20
bs2 = 100
epochn = 300

print("cls start")
for ep in range(1,10):
    idx1 = np.random.permutation(n1)
    for i in range(0, n1, bs1):
        cla.cleargrads()  # classifier
        x0 = train[idx1[i:(i+bs1) if (i+bs1) < n1 else n1]]
        x = Variable(x0)
        y0 = labels[idx1[i:(i+bs1) if (i+bs1) < n1 else n1]]
        y = Variable(y0)
        cla_loss1 = cla(x, y)               ## CLS loss 1
        cla_loss1.backward()
        optimizer_cla.update()
print("cls end")

for ep in range(1,epochn):
    cls_total_loss = 0.0
    dis_total_loss = 0.0
    gen_total_loss = 0.0
    print(ep, " start")
    idx1 = np.random.permutation(n1)
    idx2 = np.random.permutation(n2)
    for i in range(0, n1, bs1):
        cla.cleargrads()  # classifier
        dis.cleargrads()  # discriminator
        gen.cleargrads()  # generator
        #--------------------
        # labeled data
        #--------------------
        x0 = train[idx1[i:(i+bs1) if (i+bs1) < n1 else n1]]
        x = Variable(x0)
        y0 = labels[idx1[i:(i+bs1) if (i+bs1) < n1 else n1]]
        y = Variable(y0)
        cla_loss1 = cla(x, y)               ## CLS loss 1
        cls_total_loss += cla_loss1.data
        cla_loss1.backward()
        optimizer_cla.update()

        y01 = y0.reshape(len(y0),1).astype(dtype='float32')
        yx0 = Variable(np.hstack([y01,x0]))
        one1 = Variable(np.ones(len(yx0)).astype(dtype='int32'))
        dis_loss2 = dis(yx0, one1)             ## DIS loss 2
        dis_total_loss += dis_loss2.data
        dis_loss2.backward()
        optimizer_dis.update()

        #--------------------
        # unlabeled data
        #--------------------
        cla.cleargrads()  # classifier
        dis.cleargrads()  # discriminator
        i2 = (i // bs1) * bs2
        x1 = unlabeled_data[idx2[i2:(i2+bs2) if (i2+bs2) < n2 else n2]]
        x2 = Variable(x1)
        yy = F.softmax(cla.fwd(x2))
        yy0 = yy.data
        yy00 = np.zeros(len(yy0)).reshape(len(yy0),1)
        for j in range(len(yy0)):
            if yy0[j,1] > yy0[j,0]:
                yy00[j,0] = 1.0
        yx1 = Variable(np.hstack([yy00,x1]).astype(dtype='float32'))
        rf01 = np.zeros(len(yy0)).reshape(len(yy0)).astype(dtype='int32')
        for j in range(len(yy0)):
            if (np.max(yy0[j]) > 0.8):
                rf01[j] = 1
        dis_loss1 = dis(yx1, rf01)            ## DIS loss 1
        dis_total_loss += dis_loss1.data

        dis_loss1.backward()
        optimizer_dis.update()
        optimizer_cla.update()

        #--------------------
        # generated data
        #--------------------
        cla.cleargrads()  # classifier
        dis.cleargrads()  # discriminator
        gen.cleargrads()  # generator
        rcls = np.random.choice([0,1], bs2)
        x3 = gen(rcls)
        yy3 = F.softmax(cla.fwd(x3))
        cla_loss6 = F.sum(F.matmul(yy3, F.transpose(yy3)))

        cls_total_loss += cla_loss6.data

        cla_loss6.backward()  ## CLS loss 1
        optimizer_cla.update()
        optimizer_gen.update()

        dis.cleargrads()  # discriminator
        gen.cleargrads()  # generator

        x3 = gen(rcls)
        yg0 =  rcls.reshape(len(rcls),1).astype(dtype='float32')
        yx3 = F.hstack([yg0,x3])
        one3 = Variable(np.ones(len(yx3)).astype(dtype='int32'))
#        dis_loss3  = dis(yx3,one3,train=False)            ## DIS loss 3
        dis_loss3  = dis(yx3,one3)            ## DIS loss 3

        dis_total_loss += dis_loss3.data

        dis_loss3.backward()
        optimizer_dis.update()
        optimizer_gen.update()

    print("dis loss ", dis_total_loss)
    print("cls loss ", cls_total_loss)
    print_test(test)

# Test ----------------------

xt = Variable(test)
yy = cla.fwd(xt)

ans = yy.data
nrow, ncol = ans.shape
ok = 0
for i in range(nrow):
    cls = np.argmax(ans[i,:])
    print(ans[i,:], cls)
    if cls == gans[i]:
        ok += 1
print(ok, "/", nrow, " = ", (ok * 1.0)/nrow)
