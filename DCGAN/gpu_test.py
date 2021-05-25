import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, Chain, cuda


class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1=L.Linear(2, 1),
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return h


model = Model()
optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
optimizer.setup(model)

gpu_device = 0
cuda.get_device(gpu_device).use()
model.to_gpu(gpu_device)
xp = cuda.cupy

x = Variable(xp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=xp.float32))
t = Variable(xp.array([[0], [1], [1], [1]], dtype=xp.float32))

for i in range(0, 3000):
    optimizer.zero_grads()
    y = model(x)
    loss = F.mean_squared_error(y, t)
    loss.backward()
    optimizer.update()

    print("loss: ", loss.data)

print(y.data)
